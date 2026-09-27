"""Thin wrapper around a local Ollama instance. No cloud calls anywhere."""
import json
import math
import requests

OLLAMA_URL = "http://localhost:11434"
CHAT_MODEL = "qwen2.5:7b"
EMBED_MODEL = "nomic-embed-text:latest"
UTIL_MODEL = "qwen3.5:0.8b"  # small/fast model for cheap structured extraction calls


def chat(messages, model=CHAT_MODEL, temperature=0.4):
    resp = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"].strip()


def embed(text: str):
    resp = requests.post(
        f"{OLLAMA_URL}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["embedding"]


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def extract_json(text: str):
    """Best-effort pull of a JSON object out of a model response."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


def extract_fact(user_msg: str, assistant_msg: str):
    """Ask a small local model whether this exchange contains a durable fact
    worth remembering long-term (preference, account detail, etc.)."""
    prompt = (
        "You extract durable facts about a customer from one support-chat exchange.\n"
        "A durable fact is something worth remembering for future conversations: "
        "a stated preference, an account/product detail, a constraint. "
        "Do NOT extract one-off chit-chat or the current problem itself.\n\n"
        f"Customer said: {user_msg}\n"
        f"Agent replied: {assistant_msg}\n\n"
        'Respond with ONLY a JSON object: {"fact": "<short first-person-free fact or null>", '
        '"category": "preference|account_info|product_info|none"}'
    )
    try:
        raw = chat([{"role": "user", "content": prompt}], model=UTIL_MODEL, temperature=0.0)
    except requests.RequestException:
        return None
    data = extract_json(raw)
    if not data:
        return None
    fact = data.get("fact")
    if not fact or str(fact).lower() == "null" or data.get("category") == "none":
        return None
    return fact, data.get("category", "other")


def summarize_episode(transcript: str):
    """Summarize a finished ticket into (issue, resolution, category, outcome)."""
    prompt = (
        "Summarize this finished customer-support ticket for a case log.\n\n"
        f"{transcript}\n\n"
        'Respond with ONLY JSON: {"issue": "...", "resolution": "...", '
        '"category": "billing|technical|shipping|other", "outcome": "resolved|unresolved"}'
    )
    raw = chat([{"role": "user", "content": prompt}], model=UTIL_MODEL, temperature=0.0)
    data = extract_json(raw)
    if not data:
        return None
    return data
