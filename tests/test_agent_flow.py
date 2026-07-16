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


def _seed_order(order_id, amount=249.0, email="dave@example.com"):
    """Put a real order in the temp DB so a lookup of it succeeds."""
    from app.db import SessionLocal, Order
    db = SessionLocal()
    db.add(Order(id=order_id, customer_email=email, status="delivered", amount=amount, item="Mechanical Keyboard"))
    db.commit()
    db.close()


def _lookup_then_ticket_script(order_id, email="dave@example.com", final="I've created the ticket."):
    """Realistic write flow the new sequencing rule requires: look the order up
    first (verifies it), THEN propose the ticket, then a closing message."""
    return [
        AIMessage(content="", tool_calls=[
            {"name": "lookup_order", "args": {"order_id": order_id}, "id": "c_lookup"}
        ]),
        AIMessage(content="", tool_calls=[
            {"name": "create_support_ticket",
             "args": {"customer_email": email, "order_id": order_id, "subject": "Missing key"},
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

    _seed_order("ORD-1003", amount=249.0, email="bob@example.com")
    fake_llm = FakeToolCallingLLM(
        _lookup_then_ticket_script("ORD-1003", email="bob@example.com", final="I've created a ticket for you.")
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
    tickets = db.query(Ticket).filter(Ticket.customer_email == "bob@example.com").all()
    db.close()
    assert len(tickets) == 1


def test_write_action_not_executed_on_reject():
    """If a human rejects the action, the ticket must NOT be created."""
    from app.agent import build_graph as bg

    _seed_order("ORD-1004", amount=249.0, email="carol@example.com")
    fake_llm = FakeToolCallingLLM(
        _lookup_then_ticket_script("ORD-1004", email="carol@example.com", final="Understood, no ticket was created.")
    )
    graph = bg(llm_client=fake_llm, checkpointer=MemorySaver())

    run_agent("I want a ticket for the wrong item", session_id="s3", graph=graph)
    result = resume_agent("s3", approved=False, graph=graph)

    assert result["pending_approval"] is None

    from app.db import SessionLocal, Ticket
    db = SessionLocal()
    tickets = db.query(Ticket).filter(Ticket.customer_email == "carol@example.com").all()
    db.close()
    assert len(tickets) == 0


def test_ticket_for_real_order_passes_existence_check_without_prior_lookup():
    """A write for a REAL order is allowed even if the model proposed it before
    looking up -- existence is verified directly at the boundary, so there's no
    second approval. (A non-existent order is still blocked; see below.)"""
    from app.agent import build_graph as bg

    _seed_order("ORD-1002", amount=249.0)
    script = [
        AIMessage(content="", tool_calls=[
            {"name": "create_support_ticket",
             "args": {"customer_email": "dave@example.com", "order_id": "ORD-1002", "subject": "Missing key"},
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

    _seed_order("ORD-1002", amount=249.0)
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
    count = db.query(Ticket).filter(Ticket.customer_email == "dave@example.com").count()
    db.close()
    assert count == 0


def test_typed_approve_during_pause_resumes_and_executes():
    from app.agent import build_graph as bg

    _seed_order("ORD-1002", amount=249.0)
    fake_llm = FakeToolCallingLLM(_lookup_then_ticket_script("ORD-1002"))
    graph = bg(llm_client=fake_llm, checkpointer=MemorySaver())

    run_agent("open a ticket for my broken keyboard", session_id="s9", graph=graph)
    result = run_agent("yes, approve it", session_id="s9", graph=graph)

    assert result["pending_approval"] is None
    assert "ticket" in result["reply"].lower()

    from app.db import SessionLocal, Ticket
    db = SessionLocal()
    count = db.query(Ticket).filter(Ticket.customer_email == "dave@example.com").count()
    db.close()
    assert count == 1


def test_typed_reject_during_pause_resumes_without_executing():
    from app.agent import build_graph as bg

    _seed_order("ORD-1002", amount=249.0)
    fake_llm = FakeToolCallingLLM(
        _lookup_then_ticket_script("ORD-1002", final="Understood, I won't create the ticket.")
    )
    graph = bg(llm_client=fake_llm, checkpointer=MemorySaver())

    run_agent("open a ticket for my broken keyboard", session_id="s10", graph=graph)
    result = run_agent("no, cancel that", session_id="s10", graph=graph)

    assert result["pending_approval"] is None

    from app.db import SessionLocal, Ticket
    db = SessionLocal()
    count = db.query(Ticket).filter(Ticket.customer_email == "dave@example.com").count()
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

    _seed_order("ORD-1003", amount=899.0, email="bob@example.com")
    fake_llm = FakeToolCallingLLM(_lookup_then_ticket_script("ORD-1003", email="bob@example.com"))
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
    n = db.query(Ticket).filter(Ticket.customer_email == "bob@example.com").count()
    db.close()
    assert n == 1  # ticket still created per policy


def test_incidental_affirmative_does_not_approve_a_pending_write():
    """A new question that happens to contain "ok"/"yes" must NOT execute the
    pending write -- it's a message, not a verdict."""
    from app.agent import build_graph as bg

    _seed_order("ORD-1002", amount=249.0, email="dave@example.com")
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
    n = db.query(Ticket).filter(Ticket.customer_email == "dave@example.com").count()
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
