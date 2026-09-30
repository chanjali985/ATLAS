"""FastAPI backend + dashboard."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from . import llm
from .supervisor import Supervisor
from .tools import Environment

app = FastAPI(title="ATLAS — Autonomous AI Digital Twin")
STATE: dict = {}


def env() -> Environment:
    if "env" not in STATE:
        STATE["env"] = Environment()
        STATE["sup"] = Supervisor(STATE["env"])
    return STATE["env"]


def sup() -> Supervisor:
    env()
    return STATE["sup"]


class Ask(BaseModel):
    query: str


class Approval(BaseModel):
    approver: str = "operator"
    reason: str = ""


class Inject(BaseModel):
    fail_server: str | None = None
    restore_all: bool = False


@app.get("/", response_class=HTMLResponse)
def dashboard():
    return (Path(__file__).parent / "static" / "index.html").read_text()


@app.get("/api/health")
def health():
    return {"ok": True, "llm": "claude" if llm.enabled() else "offline-rules"}


@app.get("/api/state")
def state():
    e = env()
    dm = e.demand_mult()
    load = e.twin.load(dm)
    servers = [{"id": s, "rack": e.twin.servers[s].rack, "online": bool(r.online), "active": bool(r.active),
                "cpu_util": round(float(r.cpu_util), 3), "mem_util": round(float(r.mem_util), 3),
                "services": [k for k, v in e.twin.placement.items() if s in v]}
               for s, r in load.iterrows()]
    return {"summary": e.twin.evaluate(dm), "demand_mult": round(dm, 3), "servers": servers,
            "applied_changes": e.applied_changes, "forecast_eval": e.forecast_metrics}


@app.get("/api/telemetry")
def telemetry(days: int = 21):
    e = env()
    total = e.svc_df.groupby("ts").cpu_cores.sum()
    total = total[total.index >= total.index.max() - pd.Timedelta(days=days)]
    f = e.forecaster.forecast(168)
    return {"history": {"ts": [str(t) for t in total.index], "cores": [round(float(v), 2) for v in total.values]},
            "forecast": {"ts": [str(t) for t in f.index], "cores": [round(float(v), 2) for v in f.values]}}


@app.post("/api/ask")
def ask(body: Ask):
    return sup().run(body.query)


@app.get("/api/runs/{run_id}")
def get_run(run_id: str):
    try:
        return Supervisor.load(run_id)
    except FileNotFoundError:
        raise HTTPException(404, "run not found")


@app.post("/api/runs/{run_id}/resume")
def resume(run_id: str):
    return sup().run("", resume_id=run_id)


@app.post("/api/runs/{run_id}/approve")
def approve(run_id: str, body: Approval):
    try:
        return sup().approve(run_id, body.approver)
    except ValueError as e:
        raise HTTPException(409, str(e))


@app.post("/api/runs/{run_id}/reject")
def reject(run_id: str, body: Approval):
    return sup().reject(run_id, body.approver, body.reason)


@app.post("/api/inject")
def inject(body: Inject):
    e = env()
    if body.restore_all:
        for s in e.twin.servers.values():
            s.online = True
    if body.fail_server:
        if body.fail_server not in e.twin.servers:
            raise HTTPException(404, "unknown server")
        e.twin.servers[body.fail_server].online = False
        e.logs.append({"ts": str(e.svc_df.ts.max()), "service": "infra", "level": "ERROR",
                       "msg": f"{body.fail_server} unreachable: power supply fault"})
    return state()


@app.post("/api/reset")
def reset():
    STATE.clear()
    return state()
