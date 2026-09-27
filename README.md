# Agent Memory Demo — NimbusCart Support

A working demo of the four agent-memory strategies, built as a customer-service
bot. Runs entirely on local models via **Ollama** — no cloud calls anywhere.

| Strategy | What it is | Where it lives here |
|---|---|---|
| **Short-term** | Working context for the current task: objective, recent turns, pending steps. | `ShortTermMemory` in `backend/memory.py` — a plain Python dict, in-process only, keyed by `session_id`. Never touches disk. |
| **Long-term** | Knowledge that survives a single interaction: preferences, account facts. | `LongTermMemory` — SQLite table `long_term_memory`, one row per fact, each embedded with `nomic-embed-text`. Retrieved by cosine similarity only when relevant to the current message — not dumped wholesale. |
| **Episodic** | Timestamped record of past executions the system can learn from. | `EpisodicMemory` — SQLite table `episodic_memory`, one row per resolved ticket (issue + resolution + outcome + timestamp), retrieved by similarity to surface precedent. |
| **Shared** | Coordination layer for multi-agent work. | `SharedMemory` — SQLite table `blackboard`. A Triage agent, Billing agent, Tech agent, and Supervisor all read/write the same ticket's board so each sees what the others already decided. |

## Why a customer-service bot

It's the clearest setting to *feel* the difference each memory type makes:
a bot that recalls your product and shipping preference without being told
again (long-term), that recognizes "oh, this is the same class of issue we
solved last week" (episodic), that doesn't restart its train of thought every
message (short-term), and that hands a complex ticket between specialists
without the customer repeating themselves (shared).

## Setup

Requires [Ollama](https://ollama.com) running locally with these models pulled:

```bash
ollama pull qwen2.5:7b          # main chat/agent model
ollama pull nomic-embed-text    # embeddings for similarity retrieval
ollama pull qwen3.5:0.8b        # small/fast model for structured extraction
```

Then:

```bash
cd agentmemory
python3 -m venv venv && ./venv/bin/pip install -r requirements.txt
./venv/bin/uvicorn app:app --port 8008
```

Open `http://localhost:8008`. The DB (`memory.db`) is created and seeded with
a small history on first run (a demo customer's known preferences + a handful
of past resolved tickets from *other* customers) so retrieval has something to
find immediately.

## A note on model choice

The multi-agent escalation flow was originally built on `llama3.2` (3B) as
the chat model. It technically worked but the small model kept **hallucinating
that specialist resolutions had already happened** — e.g. it would read a past
episodic case and write "the replacement has been confirmed received" for a
ticket that had just been opened. Swapping the chat model to `qwen2.5:7b` fixed
this: it reliably distinguished "reference case from another customer" from
"what has actually happened on this ticket." `qwen3.5:0.8b` is still used for
the cheap, structured extraction calls (fact extraction, episode summarization,
ticket routing) where a small model is plenty and speed matters. This is worth
calling out in a presentation: **memory design doesn't fix a model that can't
reason about what's real vs. retrieved — retrieval quality and model capability
are separate concerns.**

## Demo script

1. **Short-term** — Send 2-3 messages in one session (leave "Full memory" mode
   on). Watch the *Short-term* panel accumulate turns. Refresh the page: the
   chat transcript resets but note the short-term panel is keyed by session,
   not persisted — that's the point, it's working memory, not knowledge.

2. **Long-term** — Say something like *"please always text me instead of
   emailing"*. Click **End session & log ticket**. Start a fresh message in a
   *new* browser tab (new `session_id`, same customer) and ask something that
   relates to it (e.g. *"how should I expect updates on my order?"*). The
   *Long-term* panel highlights the recalled fact in green, and the reply uses
   it without asking again.

3. **Episodic** — Ask about a problem similar to seeded history, e.g. *"my
   soundbar is making a crackling noise"*. The *Episodic* panel highlights a
   precedent case ("SoundBar Pro arrived with a cracked casing → free
   replacement"), and the agent's answer is informed by it.

4. **Shared (multi-agent)** — Type a ticket that spans two domains, e.g. *"My
   order arrived broken AND I was charged twice for it"*, then click
   **Simulate complex ticket**. Watch the *Shared / blackboard* panel fill in
   live: Triage routes to billing + technical, each specialist posts a
   resolution, the Supervisor merges both into one reply — all without the
   agents talking to each other directly, only through the shared board.

5. **Contrast** — Switch **Memory mode** to "No memory (stateless baseline)"
   and resend a message that depends on prior context. The bot re-asks for
   information it already has in Full-memory mode — the clearest before/after
   for why memory matters.

## Files

```
app.py                  FastAPI app + routes
backend/llm.py           Ollama chat/embedding client, structured extraction helpers
backend/db.py             SQLite schema + connection
backend/memory.py         ShortTermMemory, LongTermMemory, EpisodicMemory, SharedMemory
backend/agents.py         chat_turn() (single agent) and run_escalation() (4-agent pipeline)
backend/seed.py           Seeds demo history on first run
static/index.html         UI: chat + 4 live memory-inspector panels
```
