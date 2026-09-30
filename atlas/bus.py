"""Minimal A2A-style message bus: agents exchange typed messages (from, to, type, payload).
Mirrors the A2A task/message model so agents can later be split into separate services."""
from __future__ import annotations

import time
import uuid


class Bus:
    def __init__(self):
        self.messages: list[dict] = []

    def send(self, sender: str, to: str, type_: str, payload: dict) -> str:
        mid = uuid.uuid4().hex[:8]
        self.messages.append({"id": mid, "from": sender, "to": to, "type": type_,
                              "payload": payload, "t": round(time.time(), 3)})
        return mid

    def inbox(self, agent: str, type_: str | None = None) -> list[dict]:
        return [m for m in self.messages if m["to"] == agent and (type_ is None or m["type"] == type_)]
