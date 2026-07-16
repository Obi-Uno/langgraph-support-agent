import os
import tempfile
import importlib

import pytest


@pytest.fixture(autouse=True)
def temp_db(monkeypatch):
    """Point the app at a throwaway SQLite file for each test run."""
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp.name}")

    import app.db as db_module
    importlib.reload(db_module)
    db_module.init_db()

    import app.tools as tools_module
    importlib.reload(tools_module)

    yield db_module
    # Windows: dispose pooled connections so the temp DB file can be deleted
    db_module.engine.dispose()
    try:
        os.unlink(tmp.name)
    except PermissionError:
        pass


def _seed(order_id="ORD-TEST", email="x@example.com"):
    from app.db import SessionLocal, Order
    db = SessionLocal()
    db.add(Order(id=order_id, customer_email=email, status="shipped", amount=10.0, item="Widget"))
    db.commit()
    db.close()


def test_lookup_order_found(temp_db):
    import app.tools as tools_module
    _seed()

    result = tools_module.lookup_order.invoke(
        {"order_id": "ORD-TEST", "customer_email": "x@example.com"}
    )
    assert "ORD-TEST" in result
    assert "shipped" in result


def test_lookup_order_not_found(temp_db):
    import app.tools as tools_module
    result = tools_module.lookup_order.invoke(
        {"order_id": "DOES-NOT-EXIST", "customer_email": "x@example.com"}
    )
    assert "No order found" in result


def test_lookup_order_is_scoped_to_the_customer(temp_db):
    """The tool itself is scoped -- not just the graph. Someone else's order is
    indistinguishable from a missing one, and leaks nothing."""
    import app.tools as tools_module
    _seed(email="owner@example.com")

    result = tools_module.lookup_order.invoke(
        {"order_id": "ORD-TEST", "customer_email": "someone-else@example.com"}
    )
    assert result == "No order found with ID ORD-TEST."
    assert "Widget" not in result


def test_list_my_orders_only_returns_that_customers_orders(temp_db):
    import app.tools as tools_module
    _seed(order_id="ORD-MINE", email="me@example.com")
    _seed(order_id="ORD-THEIRS", email="them@example.com")

    result = tools_module.list_my_orders.invoke({"customer_email": "me@example.com"})
    assert "ORD-MINE" in result
    assert "ORD-THEIRS" not in result


def test_create_support_ticket(temp_db):
    import app.tools as tools_module
    result = tools_module.create_support_ticket.invoke({
        "customer_email": "x@example.com",
        "order_id": "ORD-TEST",
        "subject": "Item arrived broken",
    })
    assert "Created ticket" in result
