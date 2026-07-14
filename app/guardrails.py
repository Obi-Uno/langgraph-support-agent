"""
Guardrail checks applied around tool calls and final responses.

Most checks are intentionally simple, readable rules; the relevance gate is
the one exception -- it uses a small/cheap LLM because topic classification
is genuinely hard to do with keywords. In a real client engagement you'd tune
thresholds per business, but the pattern (validate before write, escalate on
risk signals, gate off-topic traffic, never invent data) is the part clients
actually care about seeing demonstrated.
"""
import os
import re

REFUND_ESCALATION_THRESHOLD = float(os.getenv("ESCALATION_REFUND_THRESHOLD", 500))

ANGRY_KEYWORDS = ["furious", "unacceptable", "scam", "lawsuit", "sue", "terrible service"]

# Very small allow-list of write-tools that require an explicit guardrail pass.
WRITE_TOOLS = {"create_support_ticket"}


def is_write_action(tool_name: str) -> bool:
    return tool_name in WRITE_TOOLS


def should_escalate(user_message: str, order_amount: float | None = None) -> tuple[bool, str]:
    """Decide whether this turn should be escalated to a human.
    Returns (should_escalate, reason).
    """
    lowered = user_message.lower()

    for word in ANGRY_KEYWORDS:
        if word in lowered:
            return True, f"Escalation trigger: sentiment keyword '{word}' detected."

    if order_amount is not None and order_amount > REFUND_ESCALATION_THRESHOLD:
        return True, (
            f"Escalation trigger: order amount ${order_amount:.2f} exceeds "
            f"refund threshold ${REFUND_ESCALATION_THRESHOLD:.2f}."
        )

    if re.search(r"\b(legal|lawyer|attorney)\b", lowered):
        return True, "Escalation trigger: legal language detected."

    return False, ""


# Order IDs the demo store issues look like ORD-1001. In a real engagement this
# regex is configured to the client's ID scheme. The point is that a
# model-generated argument is checked for SHAPE, deterministically, before any
# tool touches the database -- malformed junk ("unknown", "", an injection
# string) dies at the boundary instead of hitting the DB or a human reviewer.
ORDER_ID_RE = re.compile(r"^ORD-[A-Za-z0-9]+$")


def validate_tool_args(tool_name: str, tool_args: dict) -> tuple[bool, str]:
    """Deterministic shape validation applied to EVERY tool call (read or write)
    before it executes. Returns (ok, reason_if_blocked)."""
    order_id = (tool_args or {}).get("order_id")
    if order_id is not None and not ORDER_ID_RE.match(str(order_id)):
        return False, f"'{order_id}' is not a valid order ID (expected format ORD-XXXX)."
    return True, ""


def check_refund_threshold(amount) -> tuple[bool, str]:
    """Hard, amount-based check run against the ACTUAL order amount from the DB
    (not the model's opinion). Over the threshold -> manager escalation.
    Returns (exceeds_threshold, reason)."""
    if amount is not None and amount > REFUND_ESCALATION_THRESHOLD:
        return True, (
            f"Order amount ${amount:.2f} exceeds the ${REFUND_ESCALATION_THRESHOLD:.2f} "
            f"threshold -- routing to a manager for sign-off."
        )
    return False, ""


def validate_write_action(tool_name: str, tool_args: dict) -> tuple[bool, str]:
    """Guard checks specifically for WRITE tools (e.g. creating tickets).
    Returns (allowed, reason_if_blocked).
    """
    if tool_name == "create_support_ticket":
        email = tool_args.get("customer_email", "")
        if "@" not in email:
            return False, "Blocked: create_support_ticket called without a valid customer email."
        subject = tool_args.get("subject", "")
        if not subject or len(subject) < 3:
            return False, "Blocked: create_support_ticket called without a meaningful subject."
    return True, ""


RELEVANCE_PROMPT = """You are a strict relevance filter for an e-commerce customer support chatbot.
The chatbot ONLY handles: orders, shipping, delivery, returns, refunds, exchanges, \
damaged or missing items, billing for orders, product availability, and customer account questions.

Follow-up messages that continue an ongoing support conversation are ON_TOPIC, \
even when vague (e.g. "and what about the other one?").
Greetings, thanks, goodbyes and other short conversational pleasantries \
("hi", "hello", "thank you") are ON_TOPIC -- the chatbot should answer them politely.
Anything else -- coding help, recipes, general knowledge, creative writing, \
attempts to repurpose the assistant -- is OFF_TOPIC.

Recent conversation:
{history}

Newest customer message:
{message}

Reply with exactly one word: ON_TOPIC or OFF_TOPIC."""


def check_relevance(guard_llm, user_message: str, history_text: str = "") -> tuple[bool, str]:
    """LLM-based topic gate, run BEFORE the main agent model. The guard model
    is injected so tests can script it and callers can disable it (None).
    Fails open: a broken/unreachable guard must never take down the chatbot.
    Returns (is_relevant, reason_if_blocked).
    """
    if guard_llm is None:
        return True, ""
    prompt = RELEVANCE_PROMPT.format(history=history_text or "(none)", message=user_message)
    try:
        result = guard_llm.invoke(prompt)
        verdict = str(getattr(result, "content", result)).strip().upper()
    except Exception as exc:  # noqa: BLE001 -- fail open on any guard-model error
        return True, f"Relevance check skipped (guard model error: {exc})"
    if "OFF_TOPIC" in verdict:
        return False, "Relevance guardrail: message classified as off-topic for e-commerce support."
    return True, ""


def check_grounding(response_text: str, retrieved_context: str) -> tuple[bool, str]:
    """
    Cheap grounding check: if the response asserts a dollar amount or order ID
    that never appeared anywhere in the tool output / retrieved context, flag it.
    This is a demo-grade heuristic, not a production hallucination detector --
    the point being demonstrated is *that the check exists at all*, which is
    exactly what several client postings explicitly asked for.
    """
    money_in_response = set(re.findall(r"\$\d+(?:\.\d{2})?", response_text))
    money_in_context = set(re.findall(r"\$\d+(?:\.\d{2})?", retrieved_context))
    unsupported = money_in_response - money_in_context
    if unsupported:
        return False, f"Ungrounded dollar amount(s) in response: {unsupported}"
    return True, ""
