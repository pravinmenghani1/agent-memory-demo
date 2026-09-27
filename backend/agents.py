"""Agent logic. Two demos live here:

1. chat_turn() — a single support agent handling one message, with a
   memory_mode toggle (full / no_memory) so the SAME model visibly behaves
   differently depending on whether short/long-term/episodic memory feed it.

2. run_escalation() — a 4-agent pipeline (Triage -> Billing -> Tech ->
   Supervisor) that coordinates purely through SharedMemory (the blackboard),
   demonstrating the multi-agent "shared" memory strategy.
"""
import re
import uuid
from . import llm
from .memory import ShortTermMemory, LongTermMemory, EpisodicMemory, SharedMemory

short_term = ShortTermMemory()
long_term = LongTermMemory()
episodic = EpisodicMemory()
shared = SharedMemory()

STORE_NAME = "NimbusCart"

BASE_PERSONA = (
    f"You are Aria, a customer support agent for {STORE_NAME}, an online electronics retailer. "
    "Be warm, concise (3-5 sentences max), and never invent order/account details you were not given."
)


def chat_turn(session_id: str, user_id: str, user_msg: str, memory_mode: str = "full"):
    """memory_mode: 'full' uses all memory layers; 'none' simulates a stateless
    bot that only ever sees the current message (no short/long/episodic memory)."""
    trace = {"long_term_hits": [], "episodic_hits": [], "short_term_used": False}

    if memory_mode == "none":
        short_term.clear_session(session_id)  # stateless bot forgets everything, every turn
        messages = [
            {"role": "system", "content": BASE_PERSONA},
            {"role": "user", "content": user_msg},
        ]
        reply = llm.chat(messages)
        return reply, trace

    # --- full memory mode ---
    short_term.add_turn(session_id, "user", user_msg)
    trace["short_term_used"] = True

    lt_hits = long_term.retrieve(user_id, user_msg)
    trace["long_term_hits"] = [{"fact": h["fact"], "category": h["category"]} for h in lt_hits]

    ep_hits = episodic.retrieve_similar(user_msg)
    trace["episodic_hits"] = [
        {"issue": h["issue"], "resolution": h["resolution"], "score": h["score"]} for h in ep_hits
    ]

    context_parts = [BASE_PERSONA]
    stm_ctx = short_term.as_context_text(session_id)
    if stm_ctx:
        context_parts.append(f"[Short-term working context]\n{stm_ctx}")
    if lt_hits:
        facts = "\n".join(f"- {h['fact']}" for h in lt_hits)
        context_parts.append(f"[Long-term memory: known facts about this customer]\n{facts}")
    if ep_hits:
        cases = "\n".join(f"- Past issue: {h['issue']} -> Resolution that worked: {h['resolution']}" for h in ep_hits)
        context_parts.append(f"[Episodic memory: similar past resolved tickets]\n{cases}")
    context_parts.append(
        "Use the above memory silently to personalize your answer and avoid re-asking for "
        "information you already have. Don't say the word 'memory' to the customer."
    )
    system_msg = "\n\n".join(context_parts)

    messages = [{"role": "system", "content": system_msg}] + short_term.chat_messages(session_id)
    reply = llm.chat(messages)
    short_term.add_turn(session_id, "assistant", reply)

    # Opportunistically promote durable facts to long-term memory.
    extracted = llm.extract_fact(user_msg, reply)
    if extracted:
        fact, category = extracted
        long_term.add_fact(user_id, fact, category)
        trace["new_long_term_fact"] = {"fact": fact, "category": category}

    return reply, trace


def end_session_and_log_episode(session_id: str, user_id: str):
    """Summarize the finished conversation into an episodic memory entry."""
    snap = short_term.snapshot(session_id)
    if not snap["turns"]:
        return None
    transcript = "\n".join(f"{t['role']}: {t['content']}" for t in snap["turns"])
    summary = llm.summarize_episode(transcript)
    if not summary:
        return None
    ticket_id = f"T-{uuid.uuid4().hex[:6]}"
    episodic.log_episode(
        user_id=user_id,
        ticket_id=ticket_id,
        issue=summary.get("issue", "unknown"),
        resolution=summary.get("resolution", "unknown"),
        category=summary.get("category", "other"),
        outcome=summary.get("outcome", "resolved"),
    )
    short_term.clear_session(session_id)
    return {"ticket_id": ticket_id, **summary}


# ---------------------------------------------------------------------------
# MULTI-AGENT ESCALATION — coordination through shared memory (blackboard)
# ---------------------------------------------------------------------------
def _format_board(ticket_id):
    entries = shared.read_ticket(ticket_id)
    lines = []
    for e in entries:
        if e["agent"] == "customer":
            lines.append(f"Customer's message: {e['content']}")
        elif e["agent"] == "memory-system" and e["entry_type"] == "episodic_context":
            lines.append(
                "Reference only, from OTHER customers' past tickets - these have NOT happened "
                f"on this ticket, do not claim they already occurred here: {e['content']}"
            )
        elif e["agent"] == "memory-system":
            lines.append(f"Known fact about this customer: {e['content']}")
        else:
            lines.append(f"{e['agent'].replace('_', ' ').title()} already decided: {e['content']}")
    return "\n".join(lines)


_LEADING_TAG_RE = re.compile(r"^\s*(\[[^\]]{0,40}\]|\*\*[^*]{0,40}\*\*|[A-Za-z ]{2,25}:)\s*", re.MULTILINE)


def _clean_agent_text(text):
    # Strip stray leading bracket tags / bold headers / "Label:" prefixes the
    # small local model sometimes prepends despite instructions not to.
    prev = None
    while prev != text:
        prev = text
        text = _LEADING_TAG_RE.sub("", text, count=1).strip()
    return text


def _agent_call(role_prompt, ticket_id, extra_context=""):
    board = _format_board(ticket_id)
    system = (
        f"{role_prompt}\n\nYou are one specialist on a support team working this ticket for the "
        f"first time. Case notes so far:\n{board or '(nothing yet)'}\n\n{extra_context}\n"
        "Write 2-4 plain-prose sentences proposing what YOU will do next. This has not happened "
        "yet - propose it, don't describe it as already done.\n\n"
        "Good example style: \"I'll issue a refund for the duplicate charge and apply a goodwill "
        "credit within 3 business days.\"\n"
        "Bad example (never do this): \"[Billing Agent | Update] Refund issued.\"\n"
        "Start directly with your first sentence - no tag, no label, no heading, no brackets."
    )
    raw = llm.chat([{"role": "system", "content": system}, {"role": "user", "content": "Give your update."}])
    return _clean_agent_text(raw)


def run_escalation(user_id: str, customer_message: str):
    """Runs Triage -> [Billing, Tech as needed] -> Supervisor, all coordinating
    only via the shared blackboard. Returns the ticket_id and the full board."""
    ticket_id = f"ESC-{uuid.uuid4().hex[:6]}"
    shared.write(ticket_id, "customer", "message", customer_message)

    # Pull relevant long-term + episodic memory once, post it to the board so
    # every downstream agent sees it without re-querying.
    lt_hits = long_term.retrieve(user_id, customer_message, top_k=3, min_score=0.5)
    if lt_hits:
        shared.write(ticket_id, "memory-system", "long_term_context",
                      "; ".join(h["fact"] for h in lt_hits))
    ep_hits = episodic.retrieve_similar(customer_message, top_k=2, min_score=0.45)
    if ep_hits:
        shared.write(ticket_id, "memory-system", "episodic_context",
                      "; ".join(f"{h['issue']} -> {h['resolution']}" for h in ep_hits))

    routing_prompt = (
        "You are the Triage Agent for a customer support team.\n"
        f"Customer's message: {customer_message}\n\n"
        'Decide which specialists this ticket needs. Respond with ONLY JSON: '
        '{"billing": true|false, "technical": true|false, "reason": "<one short sentence>"}'
    )
    raw = llm.chat([{"role": "user", "content": routing_prompt}], temperature=0.0)
    routing = llm.extract_json(raw) or {}
    needs_billing = bool(routing.get("billing"))
    needs_tech = bool(routing.get("technical"))
    if not needs_billing and not needs_tech:
        needs_billing = needs_tech = True  # fail safe: route to both
    reason = routing.get("reason", "Routing based on the nature of the issue.")
    shared.write(
        ticket_id, "triage", "routing_decision",
        f"{reason} (billing: {needs_billing}, technical: {needs_tech})",
    )

    if needs_billing:
        billing_prompt = (
            "You are the Billing Agent. Resolve the billing/payment/refund part of this ticket "
            "using only what's on the case board. Propose a concrete resolution (e.g. refund amount, timeline)."
        )
        out = _agent_call(billing_prompt, ticket_id)
        shared.write(ticket_id, "billing_agent", "resolution", out)

    if needs_tech:
        tech_prompt = (
            "You are the Technical Support Agent. Resolve the technical/product/shipping-damage part "
            "of this ticket using only what's on the case board. Propose concrete next steps."
        )
        out = _agent_call(tech_prompt, ticket_id)
        shared.write(ticket_id, "tech_agent", "resolution", out)

    board = _format_board(ticket_id)
    parts_needed = [p for p, on in (("billing", needs_billing), ("technical", needs_tech)) if on]
    supervisor_system = (
        "You are the Supervisor at a customer support desk, writing the final reply to the customer.\n"
        f"Case notes so far:\n{board}\n\n"
        f"This ticket had {len(parts_needed)} part(s) handled by specialists: {', '.join(parts_needed)}. "
        "You MUST address every one of those parts in your reply - do not drop any of them, even if "
        "it makes the reply longer. Merge them into one coherent, friendly message with no contradictions. "
        "Do not mention internal agents, routing, or the case board. Start directly with your first "
        "sentence - no tag, no label, no heading, no brackets."
    )
    raw_final = llm.chat([{"role": "system", "content": supervisor_system},
                           {"role": "user", "content": "Write the final customer-facing reply now, covering every part above."}])
    final_reply = _clean_agent_text(raw_final)
    shared.write(ticket_id, "supervisor", "final_reply", final_reply)

    # Log this multi-agent resolution into episodic memory too.
    summary = llm.summarize_episode(
        f"customer: {customer_message}\nfinal_reply: {final_reply}"
    )
    if summary:
        episodic.log_episode(
            user_id=user_id, ticket_id=ticket_id,
            issue=summary.get("issue", customer_message),
            resolution=summary.get("resolution", final_reply),
            category=summary.get("category", "other"),
            outcome=summary.get("outcome", "resolved"),
        )

    return ticket_id, shared.read_ticket(ticket_id), final_reply
