"""
Tests the LangGraph plumbing itself: multi-turn memory (checkpointer), the
human-in-the-loop approval gate, and the deterministic gates around writes.

These use a scripted fake LLM rather than a live model, which is the point:
the guarantees must hold for ANY model behaviour, including adversarial. Several
tests script the model into doing the wrong thing (proposing a write for an
order it never looked up, or for one that doesn't exist) and assert the shell
stops it anyway. They need no API key and no network.
"""
import os
import tempfile

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver

from app.agent import build_graph, run_agent, resume_agent


class FakeToolCallingLLM:
    """Returns a scripted sequence of AIMessages, ignoring the actual prompt.
    .bind_tools() returns self so it slots into the same code path as a real
    chat model client.
    """

    def __init__(self, script):
        self.script = list(script)
        self.call_count = 0

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        msg = self.script[min(self.call_count, len(self.script) - 1)]
        self.call_count += 1
        return msg


@pytest.fixture(autouse=True)
def temp_db(monkeypatch):
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp.name}")

    import importlib
    import app.db as db_module
    importlib.reload(db_module)
    db_module.init_db()

    import app.tools as tools_module
    importlib.reload(tools_module)

    import app.agent as agent_module
    importlib.reload(agent_module)

    yield
    # Windows: dispose pooled connections so the temp DB file can be deleted
    db_module.engine.dispose()
    try:
        os.unlink(tmp.name)
    except PermissionError:
        pass


def _owner_of_session(session_id):
    """The customer a session is signed in as -- tests seed orders against this
    so the agent is looking at its OWN customer's data."""
    from app.identity import customer_for_session
    return customer_for_session(session_id)


def _seed_order(order_id, amount=249.0, email="dave@example.com", item="Mechanical Keyboard"):
    """Put a real order in the temp DB so a lookup of it succeeds."""
    from app.db import SessionLocal, Order
    db = SessionLocal()
    db.add(Order(id=order_id, customer_email=email, status="delivered", amount=amount, item=item))
    db.commit()
    db.close()


def _lookup_then_ticket_script(order_id, final="I've created the ticket."):
    """Realistic write flow the sequencing rule requires: look the order up first
    (verifies it), THEN propose the ticket, then a closing message.

    Note the model does not supply customer_email -- it is an InjectedToolArg and
    isn't in the schema the model sees.
    """
    return [
        AIMessage(content="", tool_calls=[
            {"name": "lookup_order", "args": {"order_id": order_id}, "id": "c_lookup"}
        ]),
        AIMessage(content="", tool_calls=[
            {"name": "create_support_ticket",
             "args": {"order_id": order_id, "subject": "Missing key"},
             "id": "c_ticket"}
        ]),
        AIMessage(content=final),
    ]


def test_read_only_tool_call_executes_without_approval():
    """lookup_order is read-only -- should NOT pause for approval."""
    from app.agent import build_graph as bg

    script = [
        AIMessage(content="", tool_calls=[
            {"name": "lookup_order", "args": {"order_id": "ORD-1001"}, "id": "call_1"}
        ]),
        AIMessage(content="Your order is on the way!"),
    ]
    fake_llm = FakeToolCallingLLM(script)
    graph = bg(llm_client=fake_llm, checkpointer=MemorySaver())

    result = run_agent("Where is my order?", session_id="s1", graph=graph)

    assert result["pending_approval"] is None
    assert "on the way" in result["reply"]


def test_reasoning_trace_logged_when_model_acts_but_not_shown_to_user():
    """ReAct 'thought': when the model calls a tool it explains why in its
    content -- that reasoning is logged as an audit event, but the intermediate
    reasoning must NOT become the user-facing reply."""
    from app.agent import build_graph as bg

    script = [
        AIMessage(
            content="The customer wants order status, so I'll look up ORD-1001.",
            tool_calls=[{"name": "lookup_order", "args": {"order_id": "ORD-1001"}, "id": "call_1"}],
        ),
        AIMessage(content="Your order is on the way!"),
    ]
    fake_llm = FakeToolCallingLLM(script)
    graph = bg(llm_client=fake_llm, checkpointer=MemorySaver())

    result = run_agent("Where is my order?", session_id="s_reason", graph=graph)

    # the reasoning is internal, not the reply
    assert result["reply"] == "Your order is on the way!"
    assert "I'll look up" not in result["reply"]

    from app.db import SessionLocal, AuditLog
    db = SessionLocal()
    reasoning_events = (
        db.query(AuditLog)
        .filter(AuditLog.session_id == "s_reason", AuditLog.event_type == "reasoning")
        .all()
    )
    db.close()
    assert len(reasoning_events) == 1
    assert "look up ORD-1001" in reasoning_events[0].detail


def test_write_action_pauses_for_approval_then_executes_on_approve():
    """create_support_ticket is a WRITE action -- must pause for a human decision."""
    from app.agent import build_graph as bg

    owner = _owner_of_session("s2")
    _seed_order("ORD-1003", amount=249.0, email=owner)
    fake_llm = FakeToolCallingLLM(
        _lookup_then_ticket_script("ORD-1003", final="I've created a ticket for you.")
    )
    graph = bg(llm_client=fake_llm, checkpointer=MemorySaver())

    first = run_agent("My chair arrived damaged", session_id="s2", graph=graph)
    assert first["pending_approval"] is not None
    assert first["pending_approval"]["tool_calls"][0]["name"] == "create_support_ticket"

    second = resume_agent("s2", approved=True, graph=graph)
    assert second["pending_approval"] is None
    assert "ticket" in second["reply"].lower()
    assert second["escalated"] is False  # $249 is under the refund threshold

    from app.db import SessionLocal, Ticket
    db = SessionLocal()
    tickets = db.query(Ticket).filter(Ticket.customer_email == owner).all()
    db.close()
    assert len(tickets) == 1  # ticket carries the SESSION's identity, not a model-supplied one


def test_write_action_not_executed_on_reject():
    """If a human rejects the action, the ticket must NOT be created."""
    from app.agent import build_graph as bg

    owner = _owner_of_session("s3")
    _seed_order("ORD-1004", amount=249.0, email=owner)
    fake_llm = FakeToolCallingLLM(
        _lookup_then_ticket_script("ORD-1004", final="Understood, no ticket was created.")
    )
    graph = bg(llm_client=fake_llm, checkpointer=MemorySaver())

    run_agent("I want a ticket for the wrong item", session_id="s3", graph=graph)
    result = resume_agent("s3", approved=False, graph=graph)

    assert result["pending_approval"] is None

    from app.db import SessionLocal, Ticket
    db = SessionLocal()
    tickets = db.query(Ticket).filter(Ticket.customer_email == owner).all()
    db.close()
    assert len(tickets) == 0


def test_ticket_for_real_order_passes_existence_check_without_prior_lookup():
    """A write for a REAL order is allowed even if the model proposed it before
    looking up -- existence is verified directly at the boundary, so there's no
    second approval. (A non-existent order is still blocked; see below.)"""
    from app.agent import build_graph as bg

    _seed_order("ORD-1002", amount=249.0, email=_owner_of_session("s11"))
    script = [
        AIMessage(content="", tool_calls=[
            {"name": "create_support_ticket",
             "args": {"order_id": "ORD-1002", "subject": "Missing key"},
             "id": "c_ticket"}
        ]),
        AIMessage(content="I've created the ticket."),
    ]
    fake_llm = FakeToolCallingLLM(script)
    graph = bg(llm_client=fake_llm, checkpointer=MemorySaver())

    run_agent("open a ticket for ORD-1002", session_id="s11", graph=graph)
    result = resume_agent("s11", approved=True, graph=graph)

    assert result["pending_approval"] is None  # single approval, no re-prompt

    from app.db import SessionLocal, Ticket
    db = SessionLocal()
    tickets = db.query(Ticket).count()
    db.close()
    assert tickets == 1


def test_ticket_blocked_for_nonexistent_order():
    """The ORD-9999 problem: a failed lookup never verifies the order, so a
    ticket for a non-existent order is blocked even after approval."""
    from app.agent import build_graph as bg

    # No seed -- the lookup will fail.
    fake_llm = FakeToolCallingLLM(_lookup_then_ticket_script("ORD-9999"))
    graph = bg(llm_client=fake_llm, checkpointer=MemorySaver())

    run_agent("refund for ORD-9999", session_id="s12", graph=graph)
    resume_agent("s12", approved=True, graph=graph)

    from app.db import SessionLocal, Ticket
    db = SessionLocal()
    tickets = db.query(Ticket).count()
    db.close()
    assert tickets == 0


class FakeGuardLLM:
    """Scripted relevance-gate verdict ("ON_TOPIC" / "OFF_TOPIC")."""

    def __init__(self, verdict):
        self.verdict = verdict

    def invoke(self, prompt):
        return AIMessage(content=self.verdict)


def test_off_topic_query_blocked_before_main_llm():
    """The relevance gate must short-circuit to a canned reply WITHOUT
    spending a main-model call."""
    from app.agent import build_graph as bg

    fake_llm = FakeToolCallingLLM([AIMessage(content="should never be reached")])
    graph = bg(llm_client=fake_llm, checkpointer=MemorySaver(), guard_llm=FakeGuardLLM("OFF_TOPIC"))

    result = run_agent("write me a python script for a todo list app", session_id="s5", graph=graph)

    assert fake_llm.call_count == 0
    assert "orders" in result["reply"].lower()
    assert result["pending_approval"] is None


def test_on_topic_query_passes_relevance_gate():
    from app.agent import build_graph as bg

    fake_llm = FakeToolCallingLLM([AIMessage(content="Your order has shipped.")])
    graph = bg(llm_client=fake_llm, checkpointer=MemorySaver(), guard_llm=FakeGuardLLM("ON_TOPIC"))

    result = run_agent("Where is order ORD-1001?", session_id="s6", graph=graph)

    assert fake_llm.call_count == 1
    assert "shipped" in result["reply"]


def test_broken_guard_model_fails_open():
    """If the guard model errors out, the query must still be answered."""
    from app.agent import build_graph as bg

    class ExplodingGuard:
        def invoke(self, prompt):
            raise RuntimeError("guard model unreachable")

    fake_llm = FakeToolCallingLLM([AIMessage(content="Happy to help with your order.")])
    graph = bg(llm_client=fake_llm, checkpointer=MemorySaver(), guard_llm=ExplodingGuard())

    result = run_agent("Where is my order?", session_id="s7", graph=graph)

    assert fake_llm.call_count == 1
    assert "order" in result["reply"].lower()


def test_unclear_message_during_pause_reminds_without_touching_graph():
    """Typing something other than approve/reject while paused must NOT
    re-invoke the graph (which would re-propose the action) -- it should
    return a reminder and keep the same pending approval."""
    from app.agent import build_graph as bg

    owner = _owner_of_session("s8")
    _seed_order("ORD-1002", amount=249.0, email=owner)
    fake_llm = FakeToolCallingLLM(_lookup_then_ticket_script("ORD-1002"))
    graph = bg(llm_client=fake_llm, checkpointer=MemorySaver())

    first = run_agent("my keyboard is missing a key, open a ticket", session_id="s8", graph=graph)
    assert first["pending_approval"] is not None
    assert fake_llm.call_count == 2  # lookup + ticket proposal

    second = run_agent("create support ticket", session_id="s8", graph=graph)
    assert second["pending_approval"] is not None
    assert "waiting for your decision" in second["reply"]
    assert fake_llm.call_count == 2  # graph was never re-invoked

    from app.db import SessionLocal, Ticket
    db = SessionLocal()
    count = db.query(Ticket).filter(Ticket.customer_email == owner).count()
    db.close()
    assert count == 0


def test_typed_approve_during_pause_resumes_and_executes():
    from app.agent import build_graph as bg

    owner = _owner_of_session("s9")
    _seed_order("ORD-1002", amount=249.0, email=owner)
    fake_llm = FakeToolCallingLLM(_lookup_then_ticket_script("ORD-1002"))
    graph = bg(llm_client=fake_llm, checkpointer=MemorySaver())

    run_agent("open a ticket for my broken keyboard", session_id="s9", graph=graph)
    result = run_agent("yes, approve it", session_id="s9", graph=graph)

    assert result["pending_approval"] is None
    assert "ticket" in result["reply"].lower()

    from app.db import SessionLocal, Ticket
    db = SessionLocal()
    count = db.query(Ticket).filter(Ticket.customer_email == owner).count()
    db.close()
    assert count == 1


def test_typed_reject_during_pause_resumes_without_executing():
    from app.agent import build_graph as bg

    owner = _owner_of_session("s10")
    _seed_order("ORD-1002", amount=249.0, email=owner)
    fake_llm = FakeToolCallingLLM(
        _lookup_then_ticket_script("ORD-1002", final="Understood, I won't create the ticket.")
    )
    graph = bg(llm_client=fake_llm, checkpointer=MemorySaver())

    run_agent("open a ticket for my broken keyboard", session_id="s10", graph=graph)
    result = run_agent("no, cancel that", session_id="s10", graph=graph)

    assert result["pending_approval"] is None

    from app.db import SessionLocal, Ticket
    db = SessionLocal()
    count = db.query(Ticket).filter(Ticket.customer_email == owner).count()
    db.close()
    assert count == 0


def test_malformed_order_id_blocked_at_tool_boundary():
    """A model-generated malformed order_id is refused before touching the DB;
    the agent still recovers and replies."""
    from app.agent import build_graph as bg

    script = [
        AIMessage(content="", tool_calls=[
            {"name": "lookup_order", "args": {"order_id": "unknown"}, "id": "c1"}
        ]),
        AIMessage(content="Could you share your order number, e.g. ORD-1001?"),
    ]
    fake_llm = FakeToolCallingLLM(script)
    graph = bg(llm_client=fake_llm, checkpointer=MemorySaver())

    result = run_agent("where is my stuff", session_id="s14", graph=graph)

    assert result["pending_approval"] is None
    assert "order number" in result["reply"].lower()

    from app.db import SessionLocal, AuditLog
    db = SessionLocal()
    blocks = db.query(AuditLog).filter(AuditLog.event_type == "guardrail_block").count()
    db.close()
    assert blocks >= 1


def test_high_value_ticket_escalates_after_approval():
    """Amount-based threshold enforced in code: a ticket for an over-threshold
    order escalates to a manager regardless of the model, but is still created."""
    from app.agent import build_graph as bg

    owner = _owner_of_session("s15")
    _seed_order("ORD-1003", amount=899.0, email=owner, item="Office Chair")
    fake_llm = FakeToolCallingLLM(_lookup_then_ticket_script("ORD-1003"))
    graph = bg(llm_client=fake_llm, checkpointer=MemorySaver())

    run_agent("my chair is damaged, I want a refund", session_id="s15", graph=graph)
    result = resume_agent("s15", approved=True, graph=graph)

    assert result["escalated"] is True
    # the customer must actually be TOLD they're being escalated -- the flag
    # alone is invisible to them. This escalation is only raised during the
    # resumed run, so resume_agent has to apply the override too.
    assert "specialist" in result["reply"].lower()
    assert "899" in result["reply"]  # the threshold reason reaches the customer

    from app.db import SessionLocal, Ticket
    db = SessionLocal()
    n = db.query(Ticket).filter(Ticket.customer_email == owner).count()
    db.close()
    assert n == 1  # ticket still created per policy


def test_incidental_affirmative_does_not_approve_a_pending_write():
    """A new question that happens to contain "ok"/"yes" must NOT execute the
    pending write -- it's a message, not a verdict."""
    from app.agent import build_graph as bg

    owner = _owner_of_session("s16")
    _seed_order("ORD-1002", amount=249.0, email=owner)
    fake_llm = FakeToolCallingLLM(_lookup_then_ticket_script("ORD-1002"))
    graph = bg(llm_client=fake_llm, checkpointer=MemorySaver())

    first = run_agent("my keyboard is missing a key, open a ticket", session_id="s16", graph=graph)
    assert first["pending_approval"] is not None
    calls_at_pause = fake_llm.call_count

    second = run_agent("ok but what about ORD-1002?", session_id="s16", graph=graph)

    assert second["pending_approval"] is not None       # still waiting
    assert "waiting for your decision" in second["reply"]
    assert fake_llm.call_count == calls_at_pause        # graph untouched

    from app.db import SessionLocal, Ticket
    db = SessionLocal()
    n = db.query(Ticket).filter(Ticket.customer_email == owner).count()
    db.close()
    assert n == 0  # nothing was written


def test_decision_classifier_boundaries():
    """Pure verdicts resolve; anything carrying extra meaning asks again."""
    from app.agent import _classify_decision as c

    assert c("approve") == "approve"
    assert c("yes") == "approve"
    assert c("yes, approve it") == "approve"
    assert c("go ahead") == "approve"
    assert c("please approve that") == "approve"
    assert c("reject") == "reject"
    assert c("no, cancel that") == "reject"

    # not verdicts -- must return None so the user is asked again
    assert c("ok but what about ORD-1002?") is None
    assert c("yes and also where is my other order") is None
    assert c("no wait, yes approve it") is None   # both classes
    assert c("ok?") is None                       # a question
    assert c("create support ticket") is None
    assert c("") is None


def _other_customer_than(email):
    from app.identity import KNOWN_CUSTOMERS
    return next(c for c in KNOWN_CUSTOMERS if c != email)


def test_cannot_read_another_customers_order():
    """The core authorization guarantee: an order belonging to someone else is
    reported exactly like one that does not exist -- no data, no confirmation
    that the id is even real."""
    from app.agent import build_graph as bg

    stranger = _other_customer_than(_owner_of_session("s17"))
    _seed_order("ORD-7777", amount=899.0, email=stranger, item="Office Chair")

    script = [
        AIMessage(content="", tool_calls=[
            {"name": "lookup_order", "args": {"order_id": "ORD-7777"}, "id": "c1"}
        ]),
        AIMessage(content="I couldn't find that order."),
    ]
    fake_llm = FakeToolCallingLLM(script)
    graph = bg(llm_client=fake_llm, checkpointer=MemorySaver())

    run_agent("what is order ORD-7777?", session_id="s17", graph=graph)

    from app.db import SessionLocal, AuditLog
    db = SessionLocal()
    tool_events = db.query(AuditLog).filter(AuditLog.event_type == "tool_call").all()
    db.close()
    detail = " ".join(e.detail for e in tool_events)
    assert "No order found with ID ORD-7777" in detail
    assert "Office Chair" not in detail  # the item never leaked
    assert "899" not in detail           # nor the amount


def test_model_cannot_widen_its_own_access():
    """Even if the model somehow emits customer_email (it isn't in the schema it
    sees), execute_tools overwrites it with the session's identity."""
    from app.agent import build_graph as bg

    stranger = _other_customer_than(_owner_of_session("s18"))
    _seed_order("ORD-8888", amount=500.0, email=stranger, item="Standing Desk")

    script = [
        AIMessage(content="", tool_calls=[
            # the model tries to impersonate the order's real owner
            {"name": "lookup_order",
             "args": {"order_id": "ORD-8888", "customer_email": stranger},
             "id": "c1"}
        ]),
        AIMessage(content="I couldn't find that order."),
    ]
    fake_llm = FakeToolCallingLLM(script)
    graph = bg(llm_client=fake_llm, checkpointer=MemorySaver())

    run_agent("look up ORD-8888 for " + stranger, session_id="s18", graph=graph)

    from app.db import SessionLocal, AuditLog
    db = SessionLocal()
    tool_events = db.query(AuditLog).filter(AuditLog.event_type == "tool_call").all()
    db.close()
    detail = " ".join(e.detail for e in tool_events)
    assert "No order found with ID ORD-8888" in detail
    assert "Standing Desk" not in detail


def test_cannot_open_a_ticket_against_another_customers_order():
    """A write against someone else's order is refused by the existence gate --
    which is scoped, so 'not yours' and 'not real' are the same answer."""
    from app.agent import build_graph as bg

    stranger = _other_customer_than(_owner_of_session("s19"))
    _seed_order("ORD-6666", amount=249.0, email=stranger)

    fake_llm = FakeToolCallingLLM(_lookup_then_ticket_script("ORD-6666"))
    graph = bg(llm_client=fake_llm, checkpointer=MemorySaver())

    run_agent("open a ticket for ORD-6666", session_id="s19", graph=graph)
    resume_agent("s19", approved=True, graph=graph)

    from app.db import SessionLocal, Ticket
    db = SessionLocal()
    tickets = db.query(Ticket).count()
    db.close()
    assert tickets == 0  # approved by a human, still refused: not this customer's order


def test_list_my_orders_returns_only_the_sessions_orders():
    from app.agent import build_graph as bg

    owner = _owner_of_session("s20")
    stranger = _other_customer_than(owner)
    _seed_order("ORD-1111", amount=79.99, email=owner, item="Wireless Mouse")
    _seed_order("ORD-2222", amount=899.0, email=stranger, item="Office Chair")

    script = [
        AIMessage(content="", tool_calls=[
            {"name": "list_my_orders", "args": {}, "id": "c1"}
        ]),
        AIMessage(content="Here are your orders."),
    ]
    fake_llm = FakeToolCallingLLM(script)
    graph = bg(llm_client=fake_llm, checkpointer=MemorySaver())

    run_agent("what have I ordered?", session_id="s20", graph=graph)

    from app.db import SessionLocal, AuditLog
    db = SessionLocal()
    events = db.query(AuditLog).filter(AuditLog.event_type == "tool_call").all()
    db.close()
    detail = " ".join(e.detail for e in events)
    assert "ORD-1111" in detail
    assert "ORD-2222" not in detail  # the other customer's order is invisible


def test_policy_dollar_amounts_are_not_flagged_as_ungrounded():
    """A figure quoted from the policy docs ($15 expedited shipping) is grounded.
    ground_check must see the retrieved policy, not just tool output, or it flags
    a correct answer."""
    from app.agent import build_graph as bg

    script = [AIMessage(
        content="Standard shipping takes 3-5 business days; expedited (2-day) is an additional $15."
    )]
    fake_llm = FakeToolCallingLLM(script)
    graph = bg(llm_client=fake_llm, checkpointer=MemorySaver())

    run_agent("how long does shipping take?", session_id="s21", graph=graph)

    from app.db import SessionLocal, AuditLog
    db = SessionLocal()
    responses = (
        db.query(AuditLog)
        .filter(AuditLog.session_id == "s21", AuditLog.event_type == "response")
        .all()
    )
    db.close()
    assert responses, "expected a response event"
    assert not any("FLAGGED" in r.detail for r in responses), (
        "a $ figure that appears in the policy docs must not be flagged"
    )


def test_invented_dollar_amount_is_still_flagged():
    """The grounding check must still catch a figure backed by nothing."""
    from app.agent import build_graph as bg

    script = [AIMessage(content="I've refunded you $9999.00 for that order.")]
    fake_llm = FakeToolCallingLLM(script)
    graph = bg(llm_client=fake_llm, checkpointer=MemorySaver())

    run_agent("what about my refund?", session_id="s22", graph=graph)

    from app.db import SessionLocal, AuditLog
    db = SessionLocal()
    responses = (
        db.query(AuditLog)
        .filter(AuditLog.session_id == "s22", AuditLog.event_type == "response")
        .all()
    )
    db.close()
    assert any("FLAGGED" in r.detail and "9999" in r.detail for r in responses)


def test_tool_call_written_as_text_is_not_shown_to_the_customer():
    """llama sometimes writes a tool call as text instead of calling it. The
    tool doesn't run -- but the raw syntax must never reach the customer."""
    from app.agent import build_graph as bg

    script = [AIMessage(
        content='I need to look up the order first to confirm it exists. '
                '<function=lookup_order>{"order_id": "ORD-1003"}</function>'
    )]
    fake_llm = FakeToolCallingLLM(script)
    graph = bg(llm_client=fake_llm, checkpointer=MemorySaver())

    result = run_agent("I want to open a ticket", session_id="s23", graph=graph)

    assert "<function" not in result["reply"]
    assert "lookup_order" not in result["reply"]
    assert "order_id" not in result["reply"]
    assert result["reply"]  # still says something to the customer

    from app.db import SessionLocal, AuditLog
    db = SessionLocal()
    flagged = (
        db.query(AuditLog)
        .filter(AuditLog.session_id == "s23", AuditLog.event_type == "malformed_response")
        .count()
    )
    db.close()
    assert flagged == 1  # and it's visible in the audit trail for diagnosis


def test_greeting_is_answered_without_a_data_lookup():
    """A greeting should get a friendly reply, not the customer's order history.
    Scripted here because the guarantee we can enforce is that a no-tool reply
    reaches the customer intact -- the prompt is what keeps 'hi' out of the tool
    loop, and this pins the expected shape."""
    from app.agent import build_graph as bg

    fake_llm = FakeToolCallingLLM([AIMessage(content="Hi! How can I help you today?")])
    graph = bg(llm_client=fake_llm, checkpointer=MemorySaver())

    result = run_agent("hey", session_id="s26", graph=graph)

    assert result["reply"] == "Hi! How can I help you today?"
    assert fake_llm.call_count == 1  # no tool loop

    from app.db import SessionLocal, AuditLog
    db = SessionLocal()
    tool_calls = (
        db.query(AuditLog)
        .filter(AuditLog.session_id == "s26", AuditLog.event_type == "tool_call")
        .count()
    )
    db.close()
    assert tool_calls == 0


def test_thought_prefix_never_reaches_the_customer():
    """Asking for a rationale before tool use tempts the model to label its final
    answer "Thought: ...". Reasoning is for the audit trail, not the customer."""
    from app.agent import build_graph as bg

    script = [AIMessage(
        content="Thought: the user wants shipping times, I'll quote the policy.\n"
                "Standard shipping takes 3-5 business days."
    )]
    fake_llm = FakeToolCallingLLM(script)
    graph = bg(llm_client=fake_llm, checkpointer=MemorySaver())

    result = run_agent("how long does shipping take?", session_id="s25", graph=graph)

    assert not result["reply"].lower().startswith("thought")
    assert "Standard shipping takes 3-5 business days." in result["reply"]


def test_reply_that_is_only_tool_syntax_falls_back_gracefully():
    from app.agent import build_graph as bg

    script = [AIMessage(content='<function=lookup_order>{"order_id": "ORD-1003"}</function>')]
    fake_llm = FakeToolCallingLLM(script)
    graph = bg(llm_client=fake_llm, checkpointer=MemorySaver())

    result = run_agent("open a ticket", session_id="s24", graph=graph)

    assert "<function" not in result["reply"]
    assert "rephrase" in result["reply"].lower()  # honest fallback, not an empty bubble


def test_conversation_memory_persists_across_turns():
    """Second call with the same session_id should retain the first message
    in state -- proving the checkpointer is actually wiring up memory."""
    from app.agent import build_graph as bg

    script = [
        AIMessage(content="Sure, order ORD-1001 is shipped."),
        AIMessage(content="Yes, that's the same order we discussed."),
    ]
    fake_llm = FakeToolCallingLLM(script)
    graph = bg(llm_client=fake_llm, checkpointer=MemorySaver())

    run_agent("Tell me about ORD-1001", session_id="s4", graph=graph)
    run_agent("Is that the one I just asked about?", session_id="s4", graph=graph)

    config = {"configurable": {"thread_id": "s4"}}
    snapshot = graph.get_state(config)
    human_messages = [m.content for m in snapshot.values["messages"] if m.type == "human"]
    assert len(human_messages) == 2
    assert "ORD-1001" in human_messages[0]
