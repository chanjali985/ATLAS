"""Optional Claude integration with model routing. Falls back to deterministic logic
when ANTHROPIC_API_KEY is not set, so the whole system runs offline."""
from __future__ import annotations

import json
import os
import re

FAST_MODEL = os.getenv("ATLAS_FAST_MODEL", "claude-haiku-4-5-20251001")   # planning / routing
SMART_MODEL = os.getenv("ATLAS_SMART_MODEL", "claude-sonnet-5")           # reports / reasoning


def enabled() -> bool:
    return bool(os.getenv("ANTHROPIC_API_KEY"))


def complete(prompt: str, system: str = "", complex_task: bool = False, max_tokens: int = 1500) -> str | None:
    if not enabled():
        return None
    try:
        import anthropic
        client = anthropic.Anthropic()
        msg = client.messages.create(
            model=SMART_MODEL if complex_task else FAST_MODEL,
            max_tokens=max_tokens, system=system,
            messages=[{"role": "user", "content": prompt}])
        return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
    except Exception as e:  # never let the LLM take the system down
        print(f"[llm] falling back to rules: {e}")
        return None


def complete_json(prompt: str, system: str = "") -> dict | None:
    out = complete(prompt + "\nRespond with JSON only.", system)
    if not out:
        return None
    m = re.search(r"\{.*\}", out, re.S)
    try:
        return json.loads(m.group(0)) if m else None
    except json.JSONDecodeError:
        return None
