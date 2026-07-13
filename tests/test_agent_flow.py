"""
Tests the LangGraph plumbing itself: multi-turn memory (checkpointer) and
the human-in-the-loop approval gate on write actions.

Uses a scripted fake LLM instead of a real Anthropic call -- these tests
validate our graph wiring (routing, interrupt, resume, state persistence),
not Claude's actual reasoning, so they run with no API key and no network.
"""
import os
import tempfile

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver

from app.agent import build_graph, run_agent, resume_agent


class FakeToolCallingLLM:
    """Returns a scripted sequence of AIMessages, ignoring the actual prompt.
    Call .bind_tools() just returns self so it slots into the same code path
    as the real ChatAnthropic client.
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


def test_write_action_pauses_for_approval_then_executes_on_approve():
    """create_support_ticket is a WRITE action -- must pause for a human decision."""
    from app.agent import build_graph as bg

    script = [
        AIMessage(content="", tool_calls=[
            {
                "name": "create_support_ticket",
                "args": {"customer_email": "bob@example.com", "order_id": "ORD-1003", "subject": "Item damaged"},
                "id": "call_1",
            }
        ]),
        AIMessage(content="I've created a ticket for you."),
    ]
    fake_llm = FakeToolCallingLLM(script)
    graph = bg(llm_client=fake_llm, checkpointer=MemorySaver())

    first = run_agent("My chair arrived damaged", session_id="s2", graph=graph)
    assert first["pending_approval"] is not None
    assert first["pending_approval"]["tool_calls"][0]["name"] == "create_support_ticket"

    second = resume_agent("s2", approved=True, graph=graph)
    assert second["pending_approval"] is None
    assert "ticket" in second["reply"].lower()

    from app.db import SessionLocal, Ticket
    db = SessionLocal()
    tickets = db.query(Ticket).filter(Ticket.customer_email == "bob@example.com").all()
    db.close()
    assert len(tickets) == 1


def test_write_action_not_executed_on_reject():
    """If a human rejects the action, the ticket must NOT be created."""
    from app.agent import build_graph as bg

    script = [
        AIMessage(content="", tool_calls=[
            {
                "name": "create_support_ticket",
                "args": {"customer_email": "carol@example.com", "order_id": "ORD-1004", "subject": "Wrong item"},
                "id": "call_1",
            }
        ]),
        AIMessage(content="Understood, no ticket was created."),
    ]
    fake_llm = FakeToolCallingLLM(script)
    graph = bg(llm_client=fake_llm, checkpointer=MemorySaver())

    run_agent("I want a ticket for the wrong item", session_id="s3", graph=graph)
    result = resume_agent("s3", approved=False, graph=graph)

    assert result["pending_approval"] is None

    from app.db import SessionLocal, Ticket
    db = SessionLocal()
    tickets = db.query(Ticket).filter(Ticket.customer_email == "carol@example.com").all()
    db.close()
    assert len(tickets) == 0


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


def _write_action_script():
    return [
        AIMessage(content="", tool_calls=[
            {
                "name": "create_support_ticket",
                "args": {"customer_email": "dave@example.com", "order_id": "ORD-1002", "subject": "Missing key"},
                "id": "call_1",
            }
        ]),
        AIMessage(content="I've created the ticket."),
    ]


def test_unclear_message_during_pause_reminds_without_touching_graph():
    """Typing something other than approve/reject while paused must NOT
    re-invoke the graph (which would re-propose the action) -- it should
    return a reminder and keep the same pending approval."""
    from app.agent import build_graph as bg

    fake_llm = FakeToolCallingLLM(_write_action_script())
    graph = bg(llm_client=fake_llm, checkpointer=MemorySaver())

    first = run_agent("my keyboard is missing a key, open a ticket", session_id="s8", graph=graph)
    assert first["pending_approval"] is not None
    assert fake_llm.call_count == 1

    second = run_agent("create support ticket", session_id="s8", graph=graph)
    assert second["pending_approval"] is not None
    assert "waiting for your decision" in second["reply"]
    assert fake_llm.call_count == 1  # graph was never re-invoked

    from app.db import SessionLocal, Ticket
    db = SessionLocal()
    count = db.query(Ticket).filter(Ticket.customer_email == "dave@example.com").count()
    db.close()
    assert count == 0


def test_typed_approve_during_pause_resumes_and_executes():
    from app.agent import build_graph as bg

    fake_llm = FakeToolCallingLLM(_write_action_script())
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

    script = _write_action_script()
    script[1] = AIMessage(content="Understood, I won't create the ticket.")
    fake_llm = FakeToolCallingLLM(script)
    graph = bg(llm_client=fake_llm, checkpointer=MemorySaver())

    run_agent("open a ticket for my broken keyboard", session_id="s10", graph=graph)
    result = run_agent("no, cancel that", session_id="s10", graph=graph)

    assert result["pending_approval"] is None

    from app.db import SessionLocal, Ticket
    db = SessionLocal()
    count = db.query(Ticket).filter(Ticket.customer_email == "dave@example.com").count()
    db.close()
    assert count == 0


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
