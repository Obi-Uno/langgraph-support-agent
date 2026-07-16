"""
Customer identity for a conversation.

This is the authentication seam, and it is SIMULATED: identity is derived from
the session id, so each new visitor is signed in as a different customer and can
only ever see that customer's orders. In production this module is the only
thing that changes -- identity would come from your auth provider (a JWT subject
claim, a session cookie) instead of a hash.

What is real, and carries straight to production, is the property downstream of
here: identity is established server-side, the tools are scoped to it, and the
language model has no way to supply or override it. See tools.py, where
customer_email is an InjectedToolArg (absent from the schema the model sees) and
agent.py's execute_tools, which injects it at execution time.

Not a security boundary as written -- the session id is client-held, so a demo
visitor can pick which sample customer they are by starting a new session. That
is fine here and irrelevant to the pattern being shown.
"""
import hashlib

from app.db import SessionLocal, Order
from app.seed import SAMPLE_ORDERS

# Derived from the seed data rather than queried, so identity is deterministic
# and independent of what happens to be in the database right now. In production
# this list doesn't exist at all -- the identity arrives with the request.
KNOWN_CUSTOMERS = sorted({o["customer_email"] for o in SAMPLE_ORDERS})


def customer_for_session(session_id: str) -> str:
    """The signed-in customer for a session.

    Stable for the life of a session (so multi-turn conversations stay coherent)
    and different across sessions. Derived, never accepted from the client or the
    model.
    """
    if not KNOWN_CUSTOMERS:
        return "unknown@example.com"
    digest = hashlib.sha256(session_id.encode("utf-8")).hexdigest()
    return KNOWN_CUSTOMERS[int(digest, 16) % len(KNOWN_CUSTOMERS)]


def orders_for_customer(customer_email: str) -> list[dict]:
    """Orders visible to this customer -- used by the UI to show what they own."""
    db = SessionLocal()
    try:
        orders = (
            db.query(Order)
            .filter(Order.customer_email == customer_email)
            .order_by(Order.id)
            .all()
        )
        return [
            {"id": o.id, "item": o.item, "status": o.status, "amount": o.amount}
            for o in orders
        ]
    finally:
        db.close()
