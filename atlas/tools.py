"""Tool registry: the only way agents touch the twin, models and knowledge base.

Every call is permission-checked (guardrails) and traced (observability).
The same functions are exposed over MCP in mcp_server.py.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

from . import data, ml, optimizer
from .rag import HybridRetriever
from .twin import Twin


class ToolPermissionError(Exception):
    pass


@dataclass
class Trace:
    events: list[dict] = field(default_factory=list)

    def add(self, **kw):
        kw["t"] = round(time.time(), 3)
        self.events.append(kw)


class Environment:
    """Holds the digital twin + telemetry + trained models (built once at startup)."""

    def __init__(self, n_servers: int = 20):
        self.twin = Twin.default(n_servers)
        self.svc_df = data.generate_telemetry(self.twin)
        self.logs = data.generate_logs(self.svc_df)
        self.retriever = HybridRetriever()
        total = self.svc_df.groupby("ts").cpu_cores.sum()
        self.forecaster = ml.Forecaster()
        self.forecast_metrics = self.forecaster.fit(total)
        self.anomalies = ml.detect_anomalies(self.svc_df)
        self.applied_changes: list[dict] = []

    # current demand multiplier vs the twin's baseline (drives what-if sims)
    def demand_mult(self, window_h: int = 24 * 7) -> float:
        total = self.svc_df.groupby("ts").cpu_cores.sum()
        base = sum(s.base_cpu for s in self.twin.services.values())
        return float(total.iloc[-window_h:].mean() / base)


def build_tools(env: Environment) -> dict[str, Callable]:
    def get_metrics(window_h: int = 168) -> dict:
        m = data.server_metrics(env.twin, env.svc_df)
        recent = m[m.ts >= m.ts.max() - pd.Timedelta(hours=window_h)]
        prev = m[(m.ts < m.ts.max() - pd.Timedelta(hours=window_h)) &
                 (m.ts >= m.ts.max() - pd.Timedelta(hours=window_h + 24 * 30))]
        by_svc = env.svc_df.copy()
        cut = by_svc.ts.max() - pd.Timedelta(hours=window_h)
        now = by_svc[by_svc.ts >= cut].groupby("service").cpu_cores.mean()
        before = by_svc[(by_svc.ts < cut - pd.Timedelta(days=14)) &
                        (by_svc.ts >= cut - pd.Timedelta(days=35))].groupby("service").cpu_cores.mean()
        growth = ((now / before - 1).sort_values(ascending=False)).round(3)
        return {
            "cluster_cpu_util_recent": round(float(recent.cpu_util.mean()), 3),
            "cluster_cpu_util_prev_month": round(float(prev.cpu_util.mean()), 3),
            "cluster_cpu_change": round(float(now.sum() / before.sum() - 1), 3),
            "power_kw_avg": round(float(recent.groupby("ts").power_w.sum().mean() / 1000), 2),
            "top_service_growth": growth.head(5).to_dict(),
        }

    def detect_anomalies() -> list[dict]:
        return ml.summarize_anomalies(env.anomalies)

    def forecast_demand(horizon_h: int = 168) -> dict:
        f = env.forecaster.forecast(horizon_h)
        cap = sum(s.cpu_cores for s in env.twin.servers.values() if s.online)
        active = env.twin.evaluate(env.demand_mult())["active_servers"]
        active_cap = sum(env.twin.servers[s].cpu_cores for s, r in env.twin.load().iterrows() if r['active'])
        peak = float(f.max())
        return {"horizon_h": horizon_h, "peak_cores": round(peak, 1), "mean_cores": round(float(f.mean()), 1),
                "peak_at": str(f.idxmax()), "active_capacity_cores": active_cap, "total_capacity_cores": cap,
                "peak_vs_active_capacity": round(peak / active_cap, 3),
                "capacity_alert": peak > 0.8 * active_cap,
                "model_eval": env.forecast_metrics,
                "series": [round(float(v), 2) for v in f.values]}

    def search_logs(service: str | None = None, level: str | None = None) -> list[dict]:
        return [l for l in env.logs if (not service or l["service"] == service)
                and (not level or l["level"] == level)]

    def search_knowledge(query: str, k: int = 3) -> list[dict]:
        return env.retriever.search(query, k)

    def simulate(scenario: dict) -> dict:
        """scenario: {consolidate_to:int|None, fail_server:str|None, demand_mult:float|None, fix_etl:bool}"""
        dm = scenario.get("demand_mult") or env.demand_mult()
        if scenario.get("fix_etl"):
            dm *= scenario.get("etl_fix_factor", 0.83)
        t = env.twin
        if scenario.get("consolidate_to") is not None or scenario.get("consolidate_min"):
            t = optimizer.consolidate(t, scenario.get("consolidate_to"), dm, scenario.get("headroom", 1.0))
            if t is None:
                return {"feasible": False, "reason": "no placement satisfies constraints", "scenario": scenario}
        if scenario.get("fail_server"):
            t = t.clone(); t.servers[scenario["fail_server"]].online = False
            t = optimizer.reschedule_after_failure(t, scenario["fail_server"], dm)
        res = t.evaluate(dm)
        n1, weak = t.n_plus_one_ok(dm)
        res.update({"feasible": True, "n_plus_one": n1, "n_plus_one_weak_point": weak,
                    "demand_mult": round(dm, 3), "scenario": scenario, "placement": t.placement})
        return res

    def apply_plan(placement: dict, approved_by: str) -> dict:
        env.twin.placement = placement
        rec = {"applied_at": time.strftime("%Y-%m-%d %H:%M:%S"), "approved_by": approved_by,
               "result": env.twin.evaluate(env.demand_mult())}
        env.applied_changes.append(rec)
        return rec

    return dict(get_metrics=get_metrics, detect_anomalies=detect_anomalies,
                forecast_demand=forecast_demand, search_logs=search_logs,
                search_knowledge=search_knowledge, simulate=simulate, apply_plan=apply_plan)


class ToolBelt:
    """Per-agent, permission-scoped view of the tools."""

    def __init__(self, tools: dict, allowed: set[str], agent: str, trace: Trace):
        self._tools, self.allowed, self.agent, self.trace = tools, allowed, agent, trace

    def call(self, name: str, **kw):
        if name not in self.allowed:
            self.trace.add(kind="guardrail_block", agent=self.agent, tool=name)
            raise ToolPermissionError(f"{self.agent} may not call {name}")
        t0 = time.time()
        try:
            out = self._tools[name](**kw)
            self.trace.add(kind="tool", agent=self.agent, tool=name, args=kw,
                           ms=round((time.time() - t0) * 1000, 1), ok=True)
            return out
        except Exception as e:
            self.trace.add(kind="tool", agent=self.agent, tool=name, args=kw, ok=False, error=str(e))
            raise
