"""
Tool implementations the agent can call.

Each tool is a plain Python function whose docstring the LLM sees as the tool
description -- that docstring is how the model knows when to call it.

These stand in for a real CRM / order-management system and are the integration
seam: pointing them at a live API (Shopify, Zendesk, ...) means rewriting the
function bodies, not the agent architecture.
"""
from langchain_core.tools import tool

from app.db import SessionLocal, Order, Ticket


@tool
def lookup_order(order_id: str) -> str:
    """Look up an order by its ID and return its status, item, and amount.
    Use this whenever a customer asks about an order's status, shipping, or contents.
    """
    db = SessionLocal()
    try:
        order = db.query(Order).filter(Order.id == order_id).first()
        if not order:
            return f"No order found with ID {order_id}."
        return (
            f"Order {order.id}: item='{order.item}', status='{order.status}', "
            f"amount=${order.amount:.2f}"
        )
    finally:
        db.close()


@tool
def check_refund_status(order_id: str) -> str:
    """Check whether an order has been refunded and for how much.
    Use this when a customer asks about a refund.
    """
    db = SessionLocal()
    try:
        order = db.query(Order).filter(Order.id == order_id).first()
        if not order:
            return f"No order found with ID {order_id}."
        if order.status == "refunded":
            return f"Order {order.id} was refunded. Amount: ${order.amount:.2f}."
        return f"Order {order.id} has not been refunded. Current status: '{order.status}'."
    finally:
        db.close()


@tool
def create_support_ticket(customer_email: str, order_id: str, subject: str) -> str:
    """Create a new support ticket for a customer issue that needs human follow-up
    or that requires tracking (e.g. damaged item, unresolved complaint).
    This is a WRITE action — only call it after guardrail checks pass.
    """
    db = SessionLocal()
    try:
        ticket = Ticket(customer_email=customer_email, order_id=order_id, subject=subject, status="open")
        db.add(ticket)
        db.commit()
        db.refresh(ticket)
        return f"Created ticket #{ticket.id} for {customer_email}: '{subject}' (status: open)."
    finally:
        db.close()


ALL_TOOLS = [lookup_order, check_refund_status, create_support_ticket]
