import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from backend.db import init_db
from backend import seed as seed_mod
from backend import agents
from backend.agents import short_term, long_term, episodic, shared

app = FastAPI(title="Agent Memory Demo")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")


@app.on_event("startup")
def startup():
    init_db()
    seed_mod.seed()


class ChatRequest(BaseModel):
    session_id: str
    user_id: str = "alex_rivera"
    message: str
    memory_mode: str = "full"  # "full" | "none"


class EscalationRequest(BaseModel):
    user_id: str = "alex_rivera"
    message: str


@app.post("/api/chat")
def api_chat(req: ChatRequest):
    reply, trace = agents.chat_turn(req.session_id, req.user_id, req.message, req.memory_mode)
    return {"reply": reply, "trace": trace}


@app.post("/api/end_session")
def api_end_session(session_id: str, user_id: str = "alex_rivera"):
    result = agents.end_session_and_log_episode(session_id, user_id)
    return {"logged": result}


@app.post("/api/escalation")
def api_escalation(req: EscalationRequest):
    ticket_id, board, final_reply = agents.run_escalation(req.user_id, req.message)
    return {"ticket_id": ticket_id, "board": board, "final_reply": final_reply}


def _strip(rows, *keys):
    return [{k: v for k, v in r.items() if k not in keys} for r in rows]


@app.get("/api/state")
def api_state(session_id: str, user_id: str = "alex_rivera"):
    """Full memory snapshot for the live inspector panels."""
    return {
        "short_term": short_term.snapshot(session_id),
        "long_term": _strip(long_term.all_for_user(user_id), "embedding"),
        "episodic": _strip(episodic.all_episodes(), "embedding"),
        "shared_recent": shared.all_tickets_recent(limit=40),
    }


@app.post("/api/reset")
def api_reset(session_id: str = None):
    init_db(reset=True)
    seed_mod.seed()
    if session_id:
        short_term.clear_session(session_id)
    return {"status": "reset"}


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
