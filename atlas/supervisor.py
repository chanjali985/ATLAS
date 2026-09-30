"""Supervisor: plan -> delegate (parallel where independent) -> reflect/replan -> safety -> decide -> await approval.

State is checkpointed after every step (durable execution): an interrupted run can be resumed.
"""
from __future__ import annotations

import json
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import llm
from .agents import (ExecutorAgent, ForecastingAgent, InvestigationAgent, MonitoringAgent,
                     OptimizationAgent, SafetyAgent, SimulationAgent)
from .bus import Bus
from .tools import Environment, Trace, build_tools

RUNS = Path(__file__).resolve().parent.parent / "runs"
RUNS.mkdir(exist_ok=True)
MEMORY = RUNS / "memory.json"

AGENTS = {c.name: c for c in (MonitoringAgent, ForecastingAgent, InvestigationAgent,
                              OptimizationAgent, SimulationAgent, SafetyAgent)}
PIPELINE = ["monitoring", "forecasting", "investigation", "optimization", "simulation", "safety"]
PARALLEL = [{"monitoring", "forecasting"}]  # independent -> run concurrently


def rule_plan(query: str) -> dict:
    q = query.lower()
    steps = set()
    if re.search(r"cost|optimi|consolidat|sav|reduce|investigat|cause|incident|why", q):
        steps |= {"monitoring", "investigation", "optimization", "simulation", "safety"}
    if re.search(r"forecast|predict|next (week|month)|capacity|demand", q):
        steps |= {"forecasting"}
    what_if = {}
    if m := re.search(r"(srv-\d+)|server (\d+)", q):
        what_if["fail_server"] = m.group(1) or f"srv-{int(m.group(2)):02d}"
    if m := re.search(r"traffic (doubles|x ?(\d+(\.\d+)?)|\+(\d+)%)", q):
        what_if["surge"] = 2.0 if m.group(1) == "doubles" else (float(m.group(2)) if m.group(2) else 1 + int(m.group(4)) / 100)
    if what_if or "what if" in q or "what happens" in q:
        steps |= {"simulation"}
    if not steps:
        steps = set(PIPELINE)
    return {"steps": [s for s in PIPELINE if s in steps], "what_if": what_if, "planner": "rules"}


def llm_plan(query: str) -> dict | None:
    out = llm.complete_json(
        f"User request: {query}\nAvailable agents in pipeline order: {PIPELINE}.\n"
        "Return {\"steps\": [subset of agents needed, in order], \"what_if\": {\"fail_server\": \"srv-NN\"|null, "
        "\"surge\": float|null}, \"rationale\": str}. Include simulation+safety whenever changes are proposed.",
        system="You are the ATLAS supervisor planner for datacenter operations.")
    if out and isinstance(out.get("steps"), list):
        out["steps"] = [s for s in PIPELINE if s in out["steps"]]
        out["what_if"] = {k: v for k, v in (out.get("what_if") or {}).items() if v}
        out["planner"] = "claude"
        return out
    return None


class Supervisor:
    def __init__(self, env: Environment):
        self.env = env
        self.tools = build_tools(env)

    # ---------- durable state ----------
    def _save(self, st: dict):
        (RUNS / f"{st['id']}.json").write_text(json.dumps(st, default=str, indent=1))

    @staticmethod
    def load(run_id: str) -> dict:
        return json.loads((RUNS / f"{run_id}.json").read_text())

    # ---------- main loop ----------
    def run(self, query: str, resume_id: str | None = None) -> dict:
        trace, bus = Trace(), Bus()
        if resume_id:
            st = self.load(resume_id)
            st["resumed"] = True
        else:
            st = {"id": uuid.uuid4().hex[:10], "query": query, "created": time.strftime("%F %T"),
                  "status": "running", "done": [], "ctx": {}, "outputs": {}, "replans": []}
            st["plan"] = llm_plan(query) or rule_plan(query)
            st["memory_hits"] = self._recall(query)
        self._save(st)
        agents = {n: AGENTS[n](self.tools, trace, bus) for n in AGENTS}
        ctx = st["ctx"]
        # re-hydrate bus with persisted monitoring suspects on resume
        if "suspect_services" in ctx:
            bus.send("monitoring", "investigation", "suspects", {"services": ctx["suspect_services"]})

        todo = [s for s in st["plan"]["steps"] if s not in st["done"]]
        if "simulation" in todo and "optimization" not in st["plan"]["steps"]:
            ctx["candidates"] = self._what_if_candidates(st["plan"]["what_if"])
        i = 0
        while i < len(todo):
            step = todo[i]
            group = next((g for g in PARALLEL if step in g), {step}) & set(todo[i:])
            with ThreadPoolExecutor(max_workers=len(group)) as ex:
                futs = {n: ex.submit(self._step, agents[n], n, ctx, st) for n in group}
                for n, f in futs.items():
                    out = f.result()
                    st["outputs"][n] = out
                    ctx.update({k: v for k, v in out.items() if k not in ("summary",)})
                    st["done"].append(n)
            todo = [s for s in todo if s not in group]
            self._reflect(st, ctx, todo)
            self._save(st)
            i = 0

        st["decision"] = self._decide(st, ctx)
        st["report"] = self._report(st)
        st["trace"] = trace.events
        st["a2a_messages"] = bus.messages
        st["status"] = "awaiting_approval" if st["decision"].get("recommended") not in (None, "baseline") else "completed"
        self._remember(st)
        self._save(st)
        return st

    def _step(self, agent, name, ctx, st, retries: int = 2):
        for attempt in range(retries + 1):
            try:
                return agent.run(ctx)
            except Exception as e:
                if attempt == retries:
                    return {"error": str(e), "summary": f"{name} failed after {retries+1} attempts: {e}"}

    def _reflect(self, st, ctx, todo):
        """Reflection: check for missing evidence and replan."""
        m = ctx.get("metrics")
        if m and m["cluster_cpu_change"] > 0.1 and not ctx.get("suspect_services") and "investigation" in todo:
            st["replans"].append("CPU up >10% but no anomalies: investigate top-growth services instead")
            ctx["suspect_services"] = list(m["top_service_growth"])[:2]
        if "safety" in st["done"] and not ctx.get("_replanned"):
            ok = [v for v in ctx["verdicts"] if v["approved"] and v["name"] != "baseline"]
            if not ok and ctx.get("candidates") and "optimization" in st["plan"]["steps"]:
                ctx["_replanned"] = True
                st["replans"].append("No candidate passed safety; retrying with gentler consolidation (17-19 servers)")
                fix = any("etl" in c["name"] for c in ctx["candidates"])
                ctx["candidates"] = [{"name": f"consolidate-{n}", "scenario": {"consolidate_to": n, "fix_etl": fix}}
                                     for n in (19, 18, 17)]
                for s in ("simulation", "safety"):
                    st["done"].remove(s); todo.append(s)

    def _what_if_candidates(self, wi: dict) -> list[dict]:
        sc = {}
        if wi.get("fail_server"):
            sc["fail_server"] = wi["fail_server"]
        if wi.get("surge"):
            sc["demand_mult"] = self.env.demand_mult() * float(wi["surge"])
        return [{"name": "baseline", "scenario": {}}, {"name": "what-if", "scenario": sc}]

    def _decide(self, st, ctx) -> dict:
        res = {r["name"]: r for r in ctx.get("results", [])}
        verdicts = ctx.get("verdicts")
        if not res:
            return {"recommended": None}
        base = res.get("baseline")
        table = []
        for name, r in res.items():
            v = next((v for v in (verdicts or []) if v["name"] == name), None)
            table.append({"name": name, "feasible": r.get("feasible"), "active_servers": r.get("active_servers"),
                          "monthly_cost": r.get("monthly_cost"), "avg_cpu_util": r.get("avg_cpu_util"),
                          "peak_cpu_util_max": r.get("peak_cpu_util_max"), "n_plus_one": r.get("n_plus_one"),
                          "safe": r.get("safe"), "approved": v["approved"] if v else None,
                          "reasons": (v or {}).get("reasons", r.get("violations", []))})
        if not verdicts:  # pure what-if
            return {"recommended": None, "comparison": table}
        ok = [t for t in table if t["approved"]]
        best = min(ok, key=lambda t: t["monthly_cost"]) if ok else None
        out = {"comparison": table, "recommended": best["name"] if best else None}
        if best and base and base.get("monthly_cost"):
            out["monthly_savings"] = round(base["monthly_cost"] - best["monthly_cost"], 2)
            out["placement"] = res[best["name"]]["placement"]
        return out

    def _report(self, st) -> str:
        lines = [f"# ATLAS report — {st['query']}", ""]
        for n in st["done"]:
            lines.append(f"- **{n}**: {st['outputs'][n].get('summary')}")
        if st["replans"]:
            lines += ["", "Replans: " + " | ".join(st["replans"])]
        d = st.get("decision", {})
        if d.get("recommended"):
            lines += ["", f"**Recommendation:** `{d['recommended']}` — saves ${d.get('monthly_savings', 0):,.0f}/month. "
                          "Requires human approval before applying."]
        facts = "\n".join(lines)
        narrative = llm.complete(
            f"Write a concise executive summary (<=180 words) of this infrastructure investigation. "
            f"Only use numbers present below.\n\n{facts}\n\nDecision table:\n{json.dumps(d.get('comparison', []))[:6000]}",
            system="You are ATLAS's reporting agent. Be precise, cite evidence, mention trade-offs.", complex_task=True)
        return facts + ("\n\n## Executive summary\n" + narrative if narrative else "")

    # ---------- long-term memory ----------
    def _recall(self, query: str) -> list[dict]:
        if not MEMORY.exists():
            return []
        mem = json.loads(MEMORY.read_text())
        words = set(re.findall(r"\w+", query.lower()))
        return [m for m in mem if len(words & set(re.findall(r"\w+", m["query"].lower()))) >= 3][-3:]

    def _remember(self, st):
        mem = json.loads(MEMORY.read_text()) if MEMORY.exists() else []
        mem.append({"run": st["id"], "query": st["query"], "recommended": st["decision"].get("recommended"),
                    "savings": st["decision"].get("monthly_savings")})
        MEMORY.write_text(json.dumps(mem[-200:], indent=1))

    # ---------- human in the loop ----------
    def approve(self, run_id: str, approver: str) -> dict:
        st = self.load(run_id)
        if st["status"] != "awaiting_approval":
            raise ValueError(f"run {run_id} is {st['status']}, not awaiting approval")
        trace = Trace()
        ex = ExecutorAgent(self.tools, trace, Bus())
        st["execution"] = ex.run({"placement": st["decision"]["placement"], "approved_by": approver})
        st["status"] = "applied"
        st["trace"] = st.get("trace", []) + trace.events
        self._save(st)
        return st

    def reject(self, run_id: str, approver: str, reason: str = "") -> dict:
        st = self.load(run_id)
        st["status"], st["rejected_by"], st["reject_reason"] = "rejected", approver, reason
        self._save(st)
        return st
