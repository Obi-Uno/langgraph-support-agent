"""
LangGraph orchestration for the support agent.

This is a ReAct agent (reason -> act -> observe, looping) built on LangGraph's
native tool-calling loop: call_model -> execute_tools -> call_model. The running
`messages` list is the agent's scratchpad, persisted by the checkpointer.

Flow per turn:
  1. Risk/escalation check + LLM relevance gate BEFORE the main model runs
     (off-topic messages get a canned reply and never reach the expensive model)
  2. Retrieve relevant policy context (RAG) and let the model reason + act. When
     it calls a tool it also emits a one-line "thought", logged as a reasoning
     event for traceability (never shown to the customer).
  3. Deterministic gates in execute_tools, enforced in code not left to the model:
     argument-shape validation, look-before-write (a ticket requires a prior
     successful lookup / existence check), and a hard refund-amount threshold.
  4. Read-only tool calls execute immediately. WRITE tool calls pause the graph
     and wait for human approval before executing -- the "AI must never
     create/update/delete records without sign-off" governance pattern.
  5. Grounding check on the final response.
  6. Every step -- reasoning, tool call, guardrail block, escalation, approval,
     response -- is logged to the audit trail.

Conversation memory: the graph is compiled with a checkpointer keyed on
session_id (LangGraph's "thread_id"), so multi-turn context (e.g. "and what
about THAT order") is preserved across separate HTTP requests, not just
within a single call.

Note: this same graph shape (retrieve -> risk-check -> tool-loop -> approval
gate -> ground-check -> log) is exactly what maps onto Google ADK's agent/tool
pattern too -- the underlying architecture is framework-agnostic.
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
from app.guardrails import (
    should_escalate, validate_write_action, validate_tool_args, is_write_action,
    check_grounding, check_relevance, check_refund_threshold,
)
from app.db import SessionLocal, log_event, Order

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
- Before creating a support ticket about an order, you MUST first call lookup_order for that order in this conversation to confirm it exists. Never call create_support_ticket for an order you have not looked up.
- If a refund or damaged-item issue needs human review, call create_support_ticket instead of promising a resolution yourself.
- When you decide to use a tool, first briefly state your reasoning in ONE short
  sentence (your "thought"), then make the tool call. This reasoning is logged
  internally for traceability -- keep it professional.
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


# Read tools that resolve an order_id against the database. A *successful* call
# to one of these proves the order exists AND that the agent looked it up --
# which is the precondition we enforce before allowing any write for that order.
READ_TOOLS_WITH_ORDER = {"lookup_order", "check_refund_status"}


def _verified_order_ids(messages) -> set:
    """Order IDs this conversation has already confirmed via a successful read.

    Deterministic: walks the message history, matches each read-tool call to
    its result by tool_call_id, and treats an order as verified only if the
    result did not report "No order found". This is what makes the
    look-before-write rule a code invariant instead of a model habit.
    """
    call_order_id = {}
    for m in messages:
        for call in (getattr(m, "tool_calls", None) or []):
            if call["name"] in READ_TOOLS_WITH_ORDER:
                oid = (call.get("args") or {}).get("order_id")
                if oid:
                    call_order_id[call["id"]] = oid

    verified = set()
    for m in messages:
        if isinstance(m, ToolMessage):
            oid = call_order_id.get(m.tool_call_id)
            if oid and "No order found" not in str(m.content):
                verified.add(oid)
    return verified


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
        """Core ReAct step. The model reads the whole scratchpad (system prompt +
        retrieved policy + running message history), reasons, and decides the next
        move: call a tool, ask the user, or answer. When it chooses to act it also
        verbalizes a one-line "thought", which we log to the audit trail for
        traceability. That thought lives on the intermediate tool-calling message,
        so it is never surfaced to the customer -- only the final answer is."""
        last_user_msg = _get_last_human_message(state)
        policy_context = retrieve_as_context(last_user_msg)
        system = SystemMessage(content=SYSTEM_PROMPT.format(policy_context=policy_context))
        response = llm_with_tools.invoke([system, *state["messages"]])
        # ReAct trace: when the model calls a tool, its content is the reasoning
        # behind that action (not a user reply) -- log it as a "reasoning" event.
        # On a final answer there are no tool calls and content IS the reply, so
        # we don't double-log it here (ground_check logs the response instead).
        if getattr(response, "tool_calls", None) and str(response.content).strip():
            db = SessionLocal()
            try:
                log_event(db, state["session_id"], "reasoning", str(response.content).strip())
            finally:
                db.close()
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
        sid = state["session_id"]
        last_ai: AIMessage = state["messages"][-1]
        tool_messages = []
        decision = state.get("pending_decision", "")
        escalated = state.get("escalated", False)
        escalation_reason = state.get("escalation_reason", "")
        verified = _verified_order_ids(state["messages"])

        # Reads before writes so that a lookup issued in the SAME batch can
        # satisfy the look-before-write rule for a write in that batch.
        calls = sorted(last_ai.tool_calls, key=lambda c: is_write_action(c["name"]))
        try:
            for call in calls:
                name, args, call_id = call["name"], call["args"], call["id"]

                # Typed boundary: validate argument SHAPE before any tool runs.
                shape_ok, shape_reason = validate_tool_args(name, args)
                if not shape_ok:
                    log_event(db, sid, "guardrail_block", shape_reason)
                    tool_messages.append(
                        ToolMessage(content=f"Action blocked by guardrail: {shape_reason}", tool_call_id=call_id)
                    )
                    continue

                if is_write_action(name):
                    allowed, reason = validate_write_action(name, args)
                    if not allowed:
                        log_event(db, sid, "guardrail_block", reason)
                        tool_messages.append(
                            ToolMessage(content=f"Action blocked by guardrail: {reason}", tool_call_id=call_id)
                        )
                        continue

                    # Existence gate: a write for an order requires that order to
                    # exist -- verified deterministically in code, not left to the
                    # model. Prefer the in-conversation lookup the agent normally
                    # does (keeps its replies grounded); fall back to a direct DB
                    # check so a real order isn't forced into a SECOND approval
                    # just because the model proposed the write before looking up.
                    # A non-existent order (the ORD-9999 case) is refused outright.
                    order_id = args.get("order_id")
                    if order_id and order_id not in verified:
                        order_exists = db.query(Order).filter(Order.id == order_id).first() is not None
                        if not order_exists:
                            reason = (
                                f"Blocked: {name} for order {order_id}, which does not exist. "
                                f"Refusing to create a ticket for an unknown order."
                            )
                            log_event(db, sid, "guardrail_block", reason)
                            tool_messages.append(
                                ToolMessage(content=f"Action blocked by guardrail: {reason}", tool_call_id=call_id)
                            )
                            continue
                        verified.add(order_id)

                    if decision != "approve":
                        decision_label = decision or "none"
                        log_event(
                            db, sid, "write_action_rejected",
                            f"{name}({args}) not executed. decision=" + repr(decision_label),
                        )
                        tool_messages.append(
                            ToolMessage(
                                content=f"Action {name!r} was not approved by the human reviewer and was not executed.",
                                tool_call_id=call_id,
                            )
                        )
                        continue

                    # Hard, amount-based threshold: read the REAL order amount
                    # from the DB and escalate over-threshold cases to a manager,
                    # regardless of what the model decided. The ticket is still
                    # created (it's the record), per policy.
                    if order_id:
                        order = db.query(Order).filter(Order.id == order_id).first()
                        if order is not None:
                            over, threshold_reason = check_refund_threshold(order.amount)
                            if over:
                                escalated = True
                                escalation_reason = threshold_reason
                                log_event(db, sid, "escalation", threshold_reason, escalated=True)

                    result = tools_by_name[name].invoke(args)
                    log_event(db, sid, "tool_call", f"{name}({args}) -> {result}")
                    tool_messages.append(ToolMessage(content=str(result), tool_call_id=call_id))
                else:
                    result = tools_by_name[name].invoke(args)
                    log_event(db, sid, "tool_call", f"{name}({args}) -> {result}")
                    order_id = args.get("order_id")
                    if order_id and "No order found" not in str(result):
                        verified.add(order_id)
                    tool_messages.append(ToolMessage(content=str(result), tool_call_id=call_id))
        finally:
            db.close()
        return {
            "messages": tool_messages,
            "pending_decision": "",
            "escalated": escalated,
            "escalation_reason": escalation_reason,
        }

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


# Vocabulary for reading a typed approve/reject on a paused session. Multi-word
# verdicts are folded into single tokens first so the closed-vocabulary check
# below can treat the message as a bag of known words.
_DECISION_PHRASES = [
    ("go ahead", "approve"),
    ("do it", "approve"),
    ("do not", "dont"),
    ("thank you", "thanks"),
]
_APPROVE_WORDS = {"approve", "approved", "yes", "yep", "yeah", "confirm", "confirmed", "okay", "ok", "sure", "proceed", "accept"}
_REJECT_WORDS = {"reject", "rejected", "no", "nope", "cancel", "deny", "denied", "dont", "don't", "stop", "abort"}
# harmless words allowed to accompany a verdict without making it ambiguous
_FILLER_WORDS = {"please", "thanks", "it", "that", "this", "now", "then", "just", "and", "the", "action", "ticket", "lets", "let's", "wait"}

PENDING_REMINDER = (
    "There's an action above waiting for your decision -- click Approve or "
    "Reject on the card, or reply 'approve' / 'reject'. I'll get to your "
    "message right after."
)


def _classify_decision(message: str):
    """Map a free-text reply on a paused session to "approve" / "reject" / None.

    This decides whether a typed message executes a pending WRITE, so it is
    deliberately strict: the message must be a decision and NOTHING else. Any
    word outside the decision/filler vocabulary -- or a question mark -- means
    this is a message, not a verdict, and we return None (the caller then just
    reminds the user to decide).

    Substring matching is not enough here: "ok but what about ORD-1002?"
    contains "ok" but is a new question, and must never approve a write. The
    only safe failure direction is toward asking again.
    """
    text = message.lower().strip()
    if "?" in text:  # a question is never a verdict
        return None
    for phrase, replacement in _DECISION_PHRASES:
        text = text.replace(phrase, replacement)
    words = re.findall(r"[a-z']+", text)
    if not words:
        return None
    # any unrecognized word means this isn't a pure decision
    if any(w not in _APPROVE_WORDS and w not in _REJECT_WORDS and w not in _FILLER_WORDS for w in words):
        return None
    approves = any(w in _APPROVE_WORDS for w in words)
    rejects = any(w in _REJECT_WORDS for w in words)
    if approves and not rejects:
        return "approve"
    if rejects and not approves:
        return "reject"
    return None  # both or neither ("no wait, yes") -> ask again


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


def _reply_for_result(result):
    """Customer-facing reply for a completed run, applying the escalation
    override. Shared by run_agent AND resume_agent: the refund-threshold
    escalation only fires after approval (inside a resumed run), so if resume
    skipped this the customer would get a plain reply while the API reported
    escalated=true."""
    reply = _final_reply_from_state(result)
    if result.get("escalated"):
        reason = result.get("escalation_reason", "")
        reply = f"I'm connecting you with a specialist for this. {reason} Someone will follow up shortly."
    return reply


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

    return {
        "session_id": session_id,
        "reply": _reply_for_result(result),
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

    return {
        "session_id": session_id,
        "reply": _reply_for_result(result),
        "escalated": result.get("escalated", False),
        "pending_approval": None,
    }
