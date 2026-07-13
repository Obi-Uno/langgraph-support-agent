"""
LangGraph orchestration for the support agent.

Flow per turn:
  1. Retrieve relevant policy context (RAG)
  2. Check sentiment/risk escalation triggers BEFORE calling the LLM
  3. Let the LLM decide + call tools (LangGraph's built-in tool loop)
  4. Read-only tool calls execute immediately. WRITE tool calls (e.g. creating
     a ticket) pause the graph and wait for human approval before executing --
     this directly mirrors the "AI must never create/update/delete records
     without sign-off" governance pattern that shows up repeatedly in
     enterprise-flavored client postings.
  5. Grounding check on the final response
  6. Log every step to the audit trail

Conversation memory: the graph is compiled with a checkpointer keyed on
session_id (LangGraph's "thread_id"), so multi-turn context (e.g. "and what
about THAT order") is preserved across separate HTTP requests, not just
within a single call.

Note: this same graph shape (retrieve -> risk-check -> tool-loop -> approval
gate -> ground-check -> log) is exactly what maps onto Google ADK's agent/tool
pattern too -- the underlying architecture is framework-agnostic, LangGraph
is just the most commonly requested keyword in client postings.
"""
import os
import re
import uuid
from typing import TypedDict, Annotated, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver

from app.tools import ALL_TOOLS
from app.rag import retrieve_as_context
from app.guardrails import should_escalate, validate_write_action, is_write_action, check_grounding, check_relevance
from app.db import SessionLocal, log_event

# LLM_PROVIDER lets you swap the model backend without touching any agent
# logic -- "gemini" or "groq" (both free-tier, good for local testing/demos)
# or "anthropic" (what production would likely run, once you have paid credits).
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()

_DEFAULT_MODELS = {
    "gemini": "gemini-2.5-flash",
    "groq": "llama-3.3-70b-versatile",
    "anthropic": "claude-haiku-4-5-20251001",
}
# `or` (not a getenv default) so the blank AGENT_MODEL= line in .env still
# falls back to the provider default.
MODEL_NAME = os.getenv("AGENT_MODEL") or _DEFAULT_MODELS.get(LLM_PROVIDER, _DEFAULT_MODELS["gemini"])

# Small/cheap models for the relevance guardrail -- a ~200 token classification
# call per chat turn. On Groq, rate limits are per model, so guard calls don't
# consume the main agent model's quota.
_GUARD_MODELS = {
    "gemini": "gemini-2.5-flash-lite",
    "groq": "llama-3.1-8b-instant",
    "anthropic": "claude-haiku-4-5-20251001",
}
GUARD_MODEL_NAME = os.getenv("GUARD_MODEL") or _GUARD_MODELS.get(LLM_PROVIDER, _GUARD_MODELS["gemini"])


def _build_default_llm():
    """Lazily imports only the provider actually selected, so you don't need
    every SDK installed/configured just to run one of them."""
    if LLM_PROVIDER == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=MODEL_NAME, temperature=0)
    if LLM_PROVIDER == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model=MODEL_NAME, temperature=0)
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0)


def _build_guard_llm():
    """Cheap model for the pre-LLM relevance gate (see guardrails.check_relevance)."""
    if LLM_PROVIDER == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=GUARD_MODEL_NAME, temperature=0)
    if LLM_PROVIDER == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model=GUARD_MODEL_NAME, temperature=0)
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(model=GUARD_MODEL_NAME, temperature=0)

def _default_checkpointer():
    """SQLite-backed graph state: sessions and paused approvals survive
    restarts. check_same_thread=False because FastAPI serves sync endpoints
    from a threadpool; SqliteSaver serializes access with its own lock."""
    import sqlite3
    conn = sqlite3.connect(os.getenv("CHECKPOINT_DB", "checkpoints.db"), check_same_thread=False)
    return SqliteSaver(conn)


SYSTEM_PROMPT = """You are a customer support agent for an e-commerce company.

Rules you MUST follow:
- Only state order/refund facts that came from a tool call result. Never guess or invent order data.
- Only state policy facts that appear in the provided policy context below.
- If a refund or damaged-item issue needs human review, call create_support_ticket instead of promising a resolution yourself.
- Be concise and friendly.

Relevant policy context for this conversation:
{policy_context}
"""


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    session_id: str
    escalated: bool
    escalation_reason: str
    pending_decision: str  # "", "approve", or "reject" -- set by a human reviewer
    off_topic: bool  # set by the relevance guardrail; routes to a canned reply


OFF_TOPIC_REPLY = (
    "I can only help with questions about your orders, shipping, returns, "
    "refunds, or your account. Could you rephrase your question about one of those?"
)


def _get_last_human_message(state: AgentState) -> str:
    return next(
        (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), ""
    )


def _has_write_tool_call(ai_message: AIMessage) -> bool:
    calls = getattr(ai_message, "tool_calls", None) or []
    return any(is_write_action(c["name"]) for c in calls)


def build_graph(llm_client=None, checkpointer=None, guard_llm=None):
    """
    llm_client: inject a fake/test LLM to test the graph without a real API
    call (see tests/test_agent_flow.py). Defaults to the real Claude client.
    checkpointer: defaults to a SQLite-backed checkpointer so conversation
    memory AND paused approvals survive server restarts. Tests inject
    MemorySaver to stay isolated and file-free.
    guard_llm: cheap model for the relevance gate. Defaults to the real guard
    client only when no fake llm_client is injected, so unit tests stay
    offline unless they explicitly script a guard.
    """
    llm = llm_client or _build_default_llm()
    llm_with_tools = llm.bind_tools(ALL_TOOLS)
    tools_by_name = {t.name: t for t in ALL_TOOLS}
    checkpointer = checkpointer or _default_checkpointer()
    if guard_llm is None and llm_client is None:
        guard_llm = _build_guard_llm()

    def retrieve_and_check(state: AgentState) -> AgentState:
        db = SessionLocal()
        try:
            last_user_msg = _get_last_human_message(state)
            escalate, reason = should_escalate(last_user_msg)
            if escalate:
                log_event(db, state["session_id"], "escalation", reason, escalated=True)
                return {"escalated": True, "escalation_reason": reason, "off_topic": False}

            history = [
                f"{'Customer' if isinstance(m, HumanMessage) else 'Agent'}: {m.content}"
                for m in state["messages"][:-1]
                if isinstance(m, (HumanMessage, AIMessage)) and m.content
            ]
            relevant, guard_reason = check_relevance(guard_llm, last_user_msg, "\n".join(history[-6:]))
            if not relevant:
                log_event(db, state["session_id"], "guardrail_block", guard_reason)
            return {"escalated": False, "escalation_reason": "", "off_topic": not relevant}
        finally:
            db.close()

    def off_topic_reply(state: AgentState) -> AgentState:
        db = SessionLocal()
        try:
            log_event(db, state["session_id"], "response", OFF_TOPIC_REPLY)
        finally:
            db.close()
        return {"messages": [AIMessage(content=OFF_TOPIC_REPLY)], "off_topic": False}

    def call_model(state: AgentState) -> AgentState:
        last_user_msg = _get_last_human_message(state)
        policy_context = retrieve_as_context(last_user_msg)
        system = SystemMessage(content=SYSTEM_PROMPT.format(policy_context=policy_context))
        response = llm_with_tools.invoke([system, *state["messages"]])
        return {"messages": [response]}

    def await_approval(state: AgentState) -> AgentState:
        db = SessionLocal()
        try:
            decision_val = state.get("pending_decision", "")
            log_event(
                db, state["session_id"], "awaiting_approval",
                "Paused for human approval. Decision received: " + repr(decision_val),
            )
        finally:
            db.close()
        return {"session_id": state["session_id"]}

    def execute_tools(state: AgentState) -> AgentState:
        db = SessionLocal()
        last_ai: AIMessage = state["messages"][-1]
        tool_messages = []
        decision = state.get("pending_decision", "")
        try:
            for call in last_ai.tool_calls:
                name, args, call_id = call["name"], call["args"], call["id"]

                if is_write_action(name):
                    allowed, reason = validate_write_action(name, args)
                    if not allowed:
                        log_event(db, state["session_id"], "guardrail_block", reason)
                        tool_messages.append(
                            ToolMessage(content=f"Action blocked by guardrail: {reason}", tool_call_id=call_id)
                        )
                        continue

                    if decision != "approve":
                        decision_label = decision or "none"
                        log_event(
                            db, state["session_id"], "write_action_rejected",
                            f"{name}({args}) not executed. decision=" + repr(decision_label),
                        )
                        tool_messages.append(
                            ToolMessage(
                                content=f"Action {name!r} was not approved by the human reviewer and was not executed.",
                                tool_call_id=call_id,
                            )
                        )
                        continue

                result = tools_by_name[name].invoke(args)
                log_event(db, state["session_id"], "tool_call", f"{name}({args}) -> {result}")
                tool_messages.append(ToolMessage(content=str(result), tool_call_id=call_id))
        finally:
            db.close()
        return {"messages": tool_messages, "pending_decision": ""}

    def ground_check(state: AgentState) -> AgentState:
        db = SessionLocal()
        try:
            final: AIMessage = state["messages"][-1]
            if isinstance(final, AIMessage) and final.content:
                context = "\n".join(
                    m.content for m in state["messages"] if isinstance(m, ToolMessage)
                )
                grounded, reason = check_grounding(str(final.content), context)
                log_event(
                    db, state["session_id"], "response",
                    str(final.content) if grounded else f"FLAGGED: {reason}",
                )
        finally:
            db.close()
        return {"session_id": state["session_id"]}

    def route_after_model(state: AgentState):
        last: AIMessage = state["messages"][-1]
        if not getattr(last, "tool_calls", None):
            return "ground_check"
        if _has_write_tool_call(last):
            return "await_approval"
        return "execute_tools"

    def route_after_check(state: AgentState):
        return "off_topic_reply" if state.get("off_topic") else "call_model"

    graph = StateGraph(AgentState)
    graph.add_node("retrieve_and_check", retrieve_and_check)
    graph.add_node("off_topic_reply", off_topic_reply)
    graph.add_node("call_model", call_model)
    graph.add_node("await_approval", await_approval)
    graph.add_node("execute_tools", execute_tools)
    graph.add_node("ground_check", ground_check)

    graph.set_entry_point("retrieve_and_check")
    graph.add_conditional_edges("retrieve_and_check", route_after_check, {
        "off_topic_reply": "off_topic_reply",
        "call_model": "call_model",
    })
    graph.add_edge("off_topic_reply", END)
    graph.add_conditional_edges("call_model", route_after_model, {
        "execute_tools": "execute_tools",
        "await_approval": "await_approval",
        "ground_check": "ground_check",
    })
    graph.add_edge("await_approval", "execute_tools")
    graph.add_edge("execute_tools", "call_model")
    graph.add_edge("ground_check", END)

    return graph.compile(checkpointer=checkpointer, interrupt_before=["await_approval"])


_compiled_graph = None


def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


def _extract_pending_approval(graph, config):
    snapshot = graph.get_state(config)
    if snapshot.next and "await_approval" in snapshot.next:
        last_ai = snapshot.values["messages"][-1]
        write_calls = [c for c in (last_ai.tool_calls or []) if is_write_action(c["name"])]
        return {"tool_calls": [{"name": c["name"], "args": c["args"]} for c in write_calls]}
    return None


_APPROVE_RE = re.compile(r"\b(approve|approved|yes|yep|yeah|confirm|go ahead|do it|okay|ok|sure)\b")
_REJECT_RE = re.compile(r"\b(reject|rejected|no|nope|cancel|deny|dont|don'?t|stop)\b")

PENDING_REMINDER = (
    "There's an action above waiting for your decision -- click Approve or "
    "Reject on the card, or reply 'approve' / 'reject'. I'll get to your "
    "message right after."
)


def _classify_decision(message: str):
    """Map a free-text reply on a paused session to approve/reject/unclear.
    Deliberately conservative: if both kinds of words appear ('no wait, yes
    approve it'), treat as unclear and ask again rather than guess."""
    lowered = message.lower()
    approves = bool(_APPROVE_RE.search(lowered))
    rejects = bool(_REJECT_RE.search(lowered))
    if approves and not rejects:
        return "approve"
    if rejects and not approves:
        return "reject"
    return None


def _log_interrupt(session_id, pending):
    """The interrupt fires BEFORE the await_approval node runs, so nothing is
    in the audit trail yet at pause time -- log the pause itself so the trail
    shows when the graph stopped and what it is waiting on."""
    db = SessionLocal()
    try:
        names = ", ".join(c["name"] for c in pending["tool_calls"])
        log_event(db, session_id, "awaiting_approval", f"Graph paused. Write action(s) pending human decision: {names}")
    finally:
        db.close()


def _final_reply_from_state(values):
    final_message = values["messages"][-1]
    return final_message.content if isinstance(final_message, AIMessage) else str(final_message.content)


def run_agent(user_message, session_id=None, graph=None):
    session_id = session_id or str(uuid.uuid4())
    graph = graph or get_graph()
    config = {"configurable": {"thread_id": session_id}}

    # If this session is paused at the approval interrupt, a new chat message
    # must NOT re-invoke the frozen graph (that would merge the message into
    # the interrupted run and re-propose the action). Instead, read the
    # message AS the decision, or remind the user a decision is pending.
    pending = _extract_pending_approval(graph, config)
    if pending:
        decision = _classify_decision(user_message)
        if decision is not None:
            return resume_agent(session_id, decision == "approve", graph)
        db = SessionLocal()
        try:
            log_event(
                db, session_id, "awaiting_approval",
                f"Message received while paused; asked for an explicit decision. Message: {user_message!r}",
            )
        finally:
            db.close()
        return {
            "session_id": session_id,
            "reply": PENDING_REMINDER,
            "escalated": False,
            "pending_approval": pending,
        }

    result = graph.invoke({
        "messages": [HumanMessage(content=user_message)],
        "session_id": session_id,
        "escalated": False,
        "escalation_reason": "",
        "pending_decision": "",
        "off_topic": False,
    }, config=config)

    pending = _extract_pending_approval(graph, config)
    if pending:
        _log_interrupt(session_id, pending)
        return {
            "session_id": session_id,
            "reply": "This action needs human approval before I can proceed. Waiting for a reviewer.",
            "escalated": False,
            "pending_approval": pending,
        }

    reply = _final_reply_from_state(result)
    if result.get("escalated"):
        reason = result["escalation_reason"]
        reply = f"I'm connecting you with a specialist for this. {reason} Someone will follow up shortly."

    return {
        "session_id": session_id,
        "reply": reply,
        "escalated": result.get("escalated", False),
        "pending_approval": None,
    }


def resume_agent(session_id, approved, graph=None):
    graph = graph or get_graph()
    config = {"configurable": {"thread_id": session_id}}

    snapshot = graph.get_state(config)
    if not snapshot.next or "await_approval" not in snapshot.next:
        return {
            "session_id": session_id,
            "reply": "No pending approval found for this session.",
            "escalated": False,
            "pending_approval": None,
        }

    graph.update_state(config, {"pending_decision": "approve" if approved else "reject"})
    result = graph.invoke(None, config=config)

    pending = _extract_pending_approval(graph, config)
    if pending:
        _log_interrupt(session_id, pending)
        return {
            "session_id": session_id,
            "reply": "Another action needs approval.",
            "escalated": False,
            "pending_approval": pending,
        }

    reply = _final_reply_from_state(result)
    return {
        "session_id": session_id,
        "reply": reply,
        "escalated": result.get("escalated", False),
        "pending_approval": None,
    }
