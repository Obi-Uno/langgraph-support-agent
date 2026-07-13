"""
Seeds the mock 'internal systems' the agent calls out to.
Run with: python -m app.seed
"""
from app.db import init_db, SessionLocal, Order, Ticket


SAMPLE_ORDERS = [
    dict(id="ORD-1001", customer_email="alice@example.com", status="shipped", amount=79.99, item="Wireless Mouse"),
    dict(id="ORD-1002", customer_email="alice@example.com", status="delivered", amount=249.00, item="Mechanical Keyboard"),
    dict(id="ORD-1003", customer_email="bob@example.com", status="placed", amount=899.00, item="Office Chair"),
    dict(id="ORD-1004", customer_email="carol@example.com", status="delivered", amount=1299.00, item="Standing Desk"),
    dict(id="ORD-1005", customer_email="bob@example.com", status="refunded", amount=45.50, item="USB-C Cable"),
]

SAMPLE_TICKETS = [
    dict(customer_email="carol@example.com", order_id="ORD-1004", subject="Desk arrived with a scratch", status="open"),
]


def run():
    init_db()
    db = SessionLocal()
    try:
        if db.query(Order).count() == 0:
            for o in SAMPLE_ORDERS:
                db.add(Order(**o))
        if db.query(Ticket).count() == 0:
            for t in SAMPLE_TICKETS:
                db.add(Ticket(**t))
        db.commit()
        print(f"Seeded {db.query(Order).count()} orders and {db.query(Ticket).count()} tickets.")
    finally:
        db.close()


if __name__ == "__main__":
    run()
