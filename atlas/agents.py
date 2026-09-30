"""Specialist agents. Each has a role, a scoped tool belt, and returns structured findings.

Agents decide WHICH tools to call and INTERPRET results; numbers come from the tools.
Agents talk to each other through an A2A-style message bus (see bus.py).
"""
from __future__ import annotations

from .bus import Bus
from .tools import ToolBelt, Trace

PERMISSIONS = {
    "monitoring":    {"get_metrics", "detect_anomalies"},
    "forecasting":   {"forecast_demand"},
    "investigation": {"search_logs", "search_knowledge"},
    "optimization":  {"simulate", "search_knowledge"},
    "simulation":    {"simulate"},
    "safety":        {"simulate", "search_knowledge"},
    "executor":      {"apply_plan"},  # only after human approval
}


class Agent:
    name = "agent"

    def __init__(self, tools: dict, trace: Trace, bus: Bus):
        self.belt = ToolBelt(tools, PERMISSIONS[self.name], self.name, trace)
        self.trace, self.bus = trace, bus

    def run(self, ctx: dict) -> dict:  # pragma: no cover
        raise NotImplementedError


class MonitoringAgent(Agent):
    name = "monitoring"

    def run(self, ctx):
        m = self.belt.call("get_metrics")
        an = self.belt.call("detect_anomalies")
        suspects = [a["service"] for a in an]
        if suspects:
            self.bus.send(self.name, "investigation", "suspects", {"services": suspects})
        return {"metrics": m, "anomalies": an, "suspect_services": suspects,
                "summary": f"Cluster CPU changed {m['cluster_cpu_change']:+.0%} vs prior month; "
                           f"anomalous services: {', '.join(suspects) or 'none'}."}


class ForecastingAgent(Agent):
    name = "forecasting"

    def run(self, ctx):
        f = self.belt.call("forecast_demand", horizon_h=ctx.get("horizon_h", 168))
        ev = f["model_eval"]
        better = ev["gbm_mape"] < ev["seasonal_naive_mape"]
        return {**f, "summary": (
            f"7-day peak demand {f['peak_cores']} cores ({f['peak_vs_active_capacity']:.0%} of active capacity) "
            f"at {f['peak_at']}. GBM MAPE {ev['gbm_mape']:.1%} vs seasonal-naive {ev['seasonal_naive_mape']:.1%}"
            f"{'' if better else ' (baseline wins; treat forecast cautiously)'}.")}


class InvestigationAgent(Agent):
    name = "investigation"

    def run(self, ctx):
        suspects = [s for m in self.bus.inbox(self.name, "suspects") for s in m["payload"]["services"]]
        suspects = suspects or ctx.get("suspect_services", [])
        findings = []
        for svc in suspects:
            logs = [l for l in self.belt.call("search_logs", service=svc) if l["level"] != "INFO" or "Deploy" in l["msg"]]
            query = f"{svc} " + " ".join(l["msg"] for l in logs)
            docs = self.belt.call("search_knowledge", query=query, k=2)
            findings.append({"service": svc, "evidence_logs": logs,
                             "similar_incidents": [{"source": d["source"], "score": d["score"]} for d in docs],
                             "likely_cause": _cause(svc, logs, docs)})
        return {"findings": findings, "summary": "; ".join(f"{f['service']}: {f['likely_cause']}" for f in findings)
                or "No suspects to investigate."}


def _cause(svc, logs, docs):
    text = " ".join(l["msg"] for l in logs).lower()
    if "cron" in text:
        return "cron schedule changed daily->hourly causing overlapping full rescans (matches INC-2025-031)"
    if "heap" in text or "oom" in text:
        return "model cache not evicting -> heap growth, GC CPU overhead, OOM kills (matches INC-2025-044)"
    return f"unclear; closest doc {docs[0]['source'] if docs else 'n/a'}"


class OptimizationAgent(Agent):
    name = "optimization"

    def run(self, ctx):
        """Propose candidate scenarios; simulation evaluates them."""
        fix_etl = any("cron" in f["likely_cause"] for f in ctx.get("findings", []))
        cands = [{"name": "baseline", "scenario": {}}]
        if fix_etl:
            cands.append({"name": "fix-etl-cron", "scenario": {"fix_etl": True}})
        for n in (18, 16, 14, 12):
            cands.append({"name": f"consolidate-max{n}{'+etl-fix' if fix_etl else ''}",
                          "scenario": {"consolidate_to": n, "fix_etl": fix_etl, "headroom": ctx.get("headroom", 1.0)}})
        cands.append({"name": "consolidate-min" + ("+etl-fix" if fix_etl else ""),
                      "scenario": {"consolidate_min": True, "fix_etl": fix_etl, "headroom": ctx.get("headroom", 1.0)}})
        return {"candidates": cands, "summary": f"Proposed {len(cands)} scenarios (etl fix: {fix_etl})."}


class SimulationAgent(Agent):
    name = "simulation"

    def run(self, ctx):
        results, seen = [], set()
        for c in ctx["candidates"]:
            r = self.belt.call("simulate", scenario=c["scenario"])
            key = (r.get("active_servers"), r.get("peak_cpu_util_max"), str(sorted((r.get("placement") or {}).items())))
            if r.get("feasible") and key in seen:
                continue  # identical outcome to an earlier scenario; skip duplicate
            seen.add(key)
            results.append({"name": c["name"], **r})
        # stress: best candidate under +20% surge and a failure of its busiest server
        return {"results": results, "summary": f"Simulated {len(results)} scenarios in the digital twin."}


class SafetyAgent(Agent):
    """Independent verifier: re-runs sims itself (does not trust the optimizer) and applies policy."""
    name = "safety"

    def run(self, ctx):
        policy = self.belt.call("search_knowledge", query="consolidation runbook constraints N+1 racks approval", k=1)
        verdicts = []
        for r in ctx["results"]:
            if not r.get("feasible"):
                verdicts.append({"name": r["name"], "approved": False, "reasons": [r.get("reason")]}); continue
            reasons = list(r["violations"])
            if not r["n_plus_one"]:
                reasons.append(f"fails N+1: losing {r['n_plus_one_weak_point']} breaks constraints")
            surge = self.belt.call("simulate", scenario={**r["scenario"], "demand_mult": r["demand_mult"] * 1.2})
            if surge.get("feasible") and not surge["safe"]:
                reasons.append("unsafe under +20% traffic surge")
            verdicts.append({"name": r["name"], "approved": not reasons, "reasons": reasons,
                             "surge_peak_cpu": surge.get("peak_cpu_util_max")})
        return {"verdicts": verdicts, "policy_source": policy[0]["source"] if policy else None,
                "summary": f"{sum(v['approved'] for v in verdicts)}/{len(verdicts)} scenarios pass safety checks."}


class ExecutorAgent(Agent):
    name = "executor"

    def run(self, ctx):
        return self.belt.call("apply_plan", placement=ctx["placement"], approved_by=ctx["approved_by"])
