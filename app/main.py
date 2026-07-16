import os
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.db import init_db, SessionLocal, AuditLog
from app.agent import run_agent, resume_agent
from app.seed import run as seed_run

app = FastAPI(
    title="Support Agent",
    description=(
        "A tool-calling customer support agent with guardrails, RAG-backed "
        "policy answers, human escalation, human-in-the-loop approval on write "
        "actions, conversation memory, an audit trail, and an n8n-compatible "
        "webhook."
    ),
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # open for the public sandbox; restrict before real traffic
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    init_db()
    seed_run()


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class PendingToolCall(BaseModel):
    name: str
    args: dict


class PendingApproval(BaseModel):
    tool_calls: list[PendingToolCall]


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    escalated: bool
    pending_approval: PendingApproval | None = None


class ApprovalRequest(BaseModel):
    approved: bool


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="message cannot be empty")
    result = run_agent(req.message, req.session_id)
    return result


@app.post("/webhook/n8n", response_model=ChatResponse)
def n8n_webhook(req: ChatRequest, x_webhook_secret: str | None = Header(default=None)):
    """
    Same agent, reachable from an n8n HTTP Request node (or any other caller).

    Machine-to-machine entry point: authenticated with a shared secret rather
    than a browser session, so the agent -- guardrails, approval gate and all --
    can sit inside a no-code automation workflow instead of behind a UI.
    """
    expected = os.getenv("N8N_WEBHOOK_SECRET")
    if expected and x_webhook_secret != expected:
        raise HTTPException(status_code=401, detail="invalid webhook secret")
    result = run_agent(req.message, req.session_id)
    return result


@app.post("/approve/{session_id}", response_model=ChatResponse)
def approve(session_id: str, req: ApprovalRequest):
    """
    Human-in-the-loop resume endpoint. When /chat returns a non-null
    pending_approval, the graph is PAUSED waiting for a reviewer -- nothing
    has been written to the database yet. Call this with approved true or
    false to resume execution.
    """
    result = resume_agent(session_id, req.approved)
    return result


@app.get("/audit-log/{session_id}")
def get_audit_log(session_id: str):
    db = SessionLocal()
    try:
        entries = (
            db.query(AuditLog)
            .filter(AuditLog.session_id == session_id)
            .order_by(AuditLog.created_at.asc())
            .all()
        )
        return [
            {
                "event_type": e.event_type,
                "detail": e.detail,
                "escalated": e.escalated,
                "created_at": e.created_at.isoformat(),
            }
            for e in entries
        ]
    finally:
        db.close()


class RevalidatingStaticFiles(StaticFiles):
    """Serve the UI with revalidation required.

    StaticFiles sends ETag/Last-Modified but no Cache-Control, so browsers fall
    back to heuristic caching and can keep serving a stale index.html for hours
    after a deploy -- which looks exactly like a broken UI. "no-cache" doesn't
    disable caching; it forces a revalidation request, which the ETag answers
    with a 304 when nothing changed. Costs a round trip, not bandwidth.
    """

    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        response.headers.setdefault("Cache-Control", "no-cache")
        return response


# Serve the chat UI at /
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
if os.path.isdir(frontend_dir):
    app.mount("/", RevalidatingStaticFiles(directory=frontend_dir, html=True), name="frontend")
