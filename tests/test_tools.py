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


def test_lookup_order_found(temp_db):
    from app.db import SessionLocal, Order
    import app.tools as tools_module

    db = SessionLocal()
    db.add(Order(id="ORD-TEST", customer_email="x@example.com", status="shipped", amount=10.0, item="Widget"))
    db.commit()
    db.close()

    result = tools_module.lookup_order.invoke({"order_id": "ORD-TEST"})
    assert "ORD-TEST" in result
    assert "shipped" in result


def test_lookup_order_not_found(temp_db):
    import app.tools as tools_module
    result = tools_module.lookup_order.invoke({"order_id": "DOES-NOT-EXIST"})
    assert "No order found" in result


def test_create_support_ticket(temp_db):
    import app.tools as tools_module
    result = tools_module.create_support_ticket.invoke({
        "customer_email": "x@example.com",
        "order_id": "ORD-TEST",
        "subject": "Item arrived broken",
    })
    assert "Created ticket" in result
