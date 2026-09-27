"""
Four memory strategies for the customer-service agent:

- ShortTermMemory: in-process, per-session conversation buffer + task scratchpad.
  Lives only in the Python process (== "the context window"). Never touches disk.
- LongTermMemory: SQLite-backed facts about a user (preferences, account info),
  retrieved by embedding similarity only when relevant to the current message.
- EpisodicMemory: SQLite-backed, timestamped log of past *resolved tickets*
  (any user), retrieved by similarity so the agent can learn from precedent.
- SharedMemory: a blackboard multiple agents read/write during one ticket, so a
  triage agent, a specialist agent, and a supervisor all see the same evolving
  state instead of re-deriving it.
"""
import json
import datetime
from collections import defaultdict, deque

from . import llm
from .db import get_conn


def now():
    return datetime.datetime.now().isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# SHORT-TERM MEMORY — working context for the current task
# ---------------------------------------------------------------------------
class ShortTermMemory:
    """Pure in-memory, per-session. Deliberately NOT persisted to disk: this is
    what the agent is holding 'in mind' right now, not what it knows forever."""

    MAX_TURNS = 12

    def __init__(self):
        self._sessions = defaultdict(lambda: {
            "objective": None,
            "turns": deque(maxlen=self.MAX_TURNS),
            "pending_steps": [],
        })

    def get(self, session_id):
        return self._sessions[session_id]

    def set_objective(self, session_id, objective):
        self._sessions[session_id]["objective"] = objective

    def add_turn(self, session_id, role, content):
        self._sessions[session_id]["turns"].append({"role": role, "content": content})

    def set_pending_steps(self, session_id, steps):
        self._sessions[session_id]["pending_steps"] = steps

    def as_context_text(self, session_id):
        s = self._sessions[session_id]
        lines = []
        if s["objective"]:
            lines.append(f"Current objective: {s['objective']}")
        if s["pending_steps"]:
            lines.append("Pending steps: " + "; ".join(s["pending_steps"]))
        return "\n".join(lines)

    def chat_messages(self, session_id):
        return [{"role": t["role"], "content": t["content"]} for t in self._sessions[session_id]["turns"]]

    def clear_session(self, session_id):
        self._sessions.pop(session_id, None)

    def snapshot(self, session_id):
        s = self._sessions[session_id]
        return {
            "objective": s["objective"],
            "pending_steps": s["pending_steps"],
            "turns": list(s["turns"]),
        }


# ---------------------------------------------------------------------------
# LONG-TERM MEMORY — knowledge that survives a single interaction
# ---------------------------------------------------------------------------
class LongTermMemory:
    TOP_K = 3
    MIN_SCORE = 0.55

    def add_fact(self, user_id, fact, category="other"):
        vec = llm.embed(fact)
        conn = get_conn()
        conn.execute(
            "INSERT INTO long_term_memory (user_id, fact, category, embedding, created_at, use_count) "
            "VALUES (?, ?, ?, ?, ?, 0)",
            (user_id, fact, category, json.dumps(vec), now()),
        )
        conn.commit()
        conn.close()

    def all_for_user(self, user_id):
        conn = get_conn()
        rows = conn.execute(
            "SELECT * FROM long_term_memory WHERE user_id = ? ORDER BY created_at DESC", (user_id,)
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def retrieve(self, user_id, query_text, top_k=None, min_score=None):
        top_k = top_k or self.TOP_K
        min_score = min_score if min_score is not None else self.MIN_SCORE
        rows = self.all_for_user(user_id)
        if not rows:
            return []
        qvec = llm.embed(query_text)
        scored = []
        for r in rows:
            sim = llm.cosine(qvec, json.loads(r["embedding"]))
            scored.append((sim, r))
        scored.sort(key=lambda x: x[0], reverse=True)
        hits = [r for sim, r in scored if sim >= min_score][:top_k]
        if hits:
            conn = get_conn()
            for r in hits:
                conn.execute(
                    "UPDATE long_term_memory SET use_count = use_count + 1, last_used_at = ? WHERE id = ?",
                    (now(), r["id"]),
                )
            conn.commit()
            conn.close()
        return hits


# ---------------------------------------------------------------------------
# EPISODIC MEMORY — timestamped record of past executions to learn from
# ---------------------------------------------------------------------------
class EpisodicMemory:
    TOP_K = 2
    MIN_SCORE = 0.5

    def log_episode(self, user_id, ticket_id, issue, resolution, category, outcome):
        vec = llm.embed(f"{issue}\n{resolution}")
        conn = get_conn()
        conn.execute(
            "INSERT INTO episodic_memory (user_id, ticket_id, timestamp, issue, resolution, category, outcome, embedding) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (user_id, ticket_id, now(), issue, resolution, category, outcome, json.dumps(vec)),
        )
        conn.commit()
        conn.close()

    def all_episodes(self):
        conn = get_conn()
        rows = conn.execute("SELECT * FROM episodic_memory ORDER BY timestamp DESC").fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def retrieve_similar(self, query_text, top_k=None, min_score=None):
        top_k = top_k or self.TOP_K
        min_score = min_score if min_score is not None else self.MIN_SCORE
        rows = self.all_episodes()
        if not rows:
            return []
        qvec = llm.embed(query_text)
        scored = [(llm.cosine(qvec, json.loads(r["embedding"])), r) for r in rows]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [dict(r, score=round(sim, 3)) for sim, r in scored if sim >= min_score][:top_k]


# ---------------------------------------------------------------------------
# SHARED MEMORY — coordination layer for multi-agent tickets (blackboard)
# ---------------------------------------------------------------------------
class SharedMemory:
    def write(self, ticket_id, agent, entry_type, content):
        conn = get_conn()
        conn.execute(
            "INSERT INTO blackboard (ticket_id, agent, entry_type, content, timestamp) VALUES (?, ?, ?, ?, ?)",
            (ticket_id, agent, entry_type, content, now()),
        )
        conn.commit()
        conn.close()

    def read_ticket(self, ticket_id):
        conn = get_conn()
        rows = conn.execute(
            "SELECT * FROM blackboard WHERE ticket_id = ? ORDER BY id ASC", (ticket_id,)
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def as_context_text(self, ticket_id):
        entries = self.read_ticket(ticket_id)
        return "\n".join(f"[{e['agent']} | {e['entry_type']}] {e['content']}" for e in entries)

    def all_tickets_recent(self, limit=50):
        conn = get_conn()
        rows = conn.execute(
            "SELECT * FROM blackboard ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]
