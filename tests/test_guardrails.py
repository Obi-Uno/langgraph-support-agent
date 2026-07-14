from app.guardrails import (
    should_escalate, validate_write_action, validate_tool_args, check_grounding,
)


def test_validate_tool_args_rejects_malformed_order_id():
    ok, reason = validate_tool_args("lookup_order", {"order_id": "unknown"})
    assert ok is False
    assert "valid order ID" in reason


def test_validate_tool_args_accepts_wellformed_order_id():
    ok, _ = validate_tool_args("lookup_order", {"order_id": "ORD-1001"})
    assert ok is True


def test_validate_tool_args_ignores_calls_without_order_id():
    ok, _ = validate_tool_args("some_tool", {"foo": "bar"})
    assert ok is True


def test_escalates_on_angry_keyword():
    escalate, reason = should_escalate("This is unacceptable, I want a refund now!")
    assert escalate is True
    assert "sentiment" in reason


def test_escalates_on_high_refund_amount():
    escalate, reason = should_escalate("Can I get a refund?", order_amount=899.00)
    assert escalate is True
    assert "threshold" in reason


def test_no_escalation_for_normal_query():
    escalate, reason = should_escalate("Where is my order ORD-1001?")
    assert escalate is False


def test_escalates_on_legal_language():
    escalate, reason = should_escalate("I am going to get my lawyer involved")
    assert escalate is True


def test_blocks_ticket_without_valid_email():
    allowed, reason = validate_write_action(
        "create_support_ticket", {"customer_email": "not-an-email", "subject": "issue"}
    )
    assert allowed is False


def test_allows_ticket_with_valid_email_and_subject():
    allowed, reason = validate_write_action(
        "create_support_ticket",
        {"customer_email": "bob@example.com", "subject": "Damaged item on arrival"},
    )
    assert allowed is True


def test_check_refund_threshold_over_and_under():
    from app.guardrails import check_refund_threshold
    over, reason = check_refund_threshold(899.00)
    assert over is True
    assert "threshold" in reason.lower()
    under, _ = check_refund_threshold(49.00)
    assert under is False


def test_grounding_flags_unsupported_dollar_amount():
    grounded, reason = check_grounding(
        "Your refund of $9999.00 has been approved.", "Order ORD-1001: amount=$79.99"
    )
    assert grounded is False


def test_grounding_passes_when_amount_matches_context():
    grounded, reason = check_grounding(
        "Your order total was $79.99.", "Order ORD-1001: amount=$79.99"
    )
    assert grounded is True
