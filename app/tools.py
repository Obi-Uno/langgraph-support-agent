"""
Tool implementations the agent can call.

Each tool is a plain Python function whose docstring the LLM sees as the tool
description -- that docstring is how the model knows when to call it.

These stand in for a real CRM / order-management system and are the integration
seam: pointing them at a live API (Shopify, Zendesk, ...) means rewriting the
function bodies, not the agent architecture.

AUTHORIZATION: every tool here is scoped to one customer. `customer_email` is an
InjectedToolArg, so it does NOT appear in the schema the model sees -- the model
cannot pass it, guess it, or be talked into changing it. agent.py's execute_tools
injects the session's identity at execution time. This is what stops the agent
from reading one customer's order on another customer's behalf: an order that
isn't yours is reported exactly as one that doesn't exist, so the agent can't be
used to probe which order ids are real.
"""
from typing import Annotated

from langchain_core.tools import tool, InjectedToolArg

from app.db import SessionLocal, Order, Ticket


@tool
def lookup_order(order_id: str, customer_email: Annotated[str, InjectedToolArg] = "") -> str:
    """Look up one of the customer's orders by its ID and return its status, item, and amount.
    Use this whenever a customer asks about an order's status, shipping, or contents.
    """
    db = SessionLocal()
    try:
        order = (
            db.query(Order)
            .filter(Order.id == order_id, Order.customer_email == customer_email)
            .first()
        )
        if not order:
            return f"No order found with ID {order_id}."
        return (
            f"Order {order.id}: item='{order.item}', status='{order.status}', "
            f"amount=${order.amount:.2f}"
        )
    finally:
        db.close()


@tool
def check_refund_status(order_id: str, customer_email: Annotated[str, InjectedToolArg] = "") -> str:
    """Check whether one of the customer's orders has been refunded and for how much.
    Use this when a customer asks about a refund.
    """
    db = SessionLocal()
    try:
        order = (
            db.query(Order)
            .filter(Order.id == order_id, Order.customer_email == customer_email)
            .first()
        )
        if not order:
            return f"No order found with ID {order_id}."
        if order.status == "refunded":
            return f"Order {order.id} was refunded. Amount: ${order.amount:.2f}."
        return f"Order {order.id} has not been refunded. Current status: '{order.status}'."
    finally:
        db.close()


@tool
def list_my_orders(customer_email: Annotated[str, InjectedToolArg] = "") -> str:
    """List all of the current customer's orders, with item, status and amount.
    Use this when the customer asks what they have ordered, or when they describe
    an issue without giving an order ID -- it is better than asking them to
    remember one.
    """
    db = SessionLocal()
    try:
        orders = (
            db.query(Order)
            .filter(Order.customer_email == customer_email)
            .order_by(Order.id)
            .all()
        )
        if not orders:
            return "This customer has no orders on file."
        lines = [
            f"Order {o.id}: item='{o.item}', status='{o.status}', amount=${o.amount:.2f}"
            for o in orders
        ]
        return "Orders for this customer:\n" + "\n".join(lines)
    finally:
        db.close()


@tool
def create_support_ticket(
    order_id: str,
    subject: str,
    customer_email: Annotated[str, InjectedToolArg] = "",
) -> str:
    """Create a support ticket for an issue that needs human follow-up or tracking
    (e.g. damaged item, unresolved complaint). Only call this for an order you
    have already looked up in this conversation.
    This is a WRITE action -- it pauses for human approval before it runs.
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


ALL_TOOLS = [lookup_order, check_refund_status, list_my_orders, create_support_ticket]

# Tools whose customer_email must be injected from the session at execution time.
IDENTITY_SCOPED_TOOLS = {t.name for t in ALL_TOOLS}
