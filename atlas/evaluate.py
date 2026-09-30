"""Evaluation harness: scores diagnosis, planning, safety and forecast quality on scripted tasks.

Run: python -m atlas.evaluate
"""
import json
import time

from .supervisor import Supervisor
from .tools import Environment

TASKS = [
    {"query": "Costs went up 30% this month. Investigate the cause and recommend a safe optimization plan.",
     "expect_causes": ["batch-etl", "recommender"], "expect_recommendation": True},
    {"query": "Forecast demand for the next 7 days and flag capacity risk.",
     "expect_steps": ["forecasting"], "expect_recommendation": False},
    {"query": "What happens if server 9 goes offline and traffic +30%?",
     "expect_whatif": {"fail_server": "srv-09"}, "expect_recommendation": False},
]


def main():
    env = Environment()
    sup = Supervisor(env)
    rows = []
    for t in TASKS:
        t0 = time.time()
        st = sup.run(t["query"])
        s = {"query": t["query"][:60], "latency_s": round(time.time() - t0, 2)}
        if "expect_causes" in t:
            found = [f["service"] for f in st["outputs"].get("investigation", {}).get("findings", [])]
            s["diagnosis_recall"] = sum(c in found for c in t["expect_causes"]) / len(t["expect_causes"])
            grounded = all(f["evidence_logs"] and f["similar_incidents"]
                           for f in st["outputs"]["investigation"]["findings"])
            s["grounded"] = grounded
        if "expect_steps" in t:
            s["plan_ok"] = all(x in st["plan"]["steps"] for x in t["expect_steps"])
        if "expect_whatif" in t:
            s["plan_ok"] = all(st["plan"]["what_if"].get(k) == v for k, v in t["expect_whatif"].items())
        rec = st.get("decision", {}).get("recommended")
        s["recommendation_ok"] = bool(rec and rec != "baseline") == t["expect_recommendation"]
        if rec:  # safety: recommended plan must independently re-verify as safe
            r = next(r for r in st["decision"]["comparison"] if r["name"] == rec)
            s["recommended_safe"] = bool(r["safe"] and r["n_plus_one"])
            s["monthly_savings"] = st["decision"].get("monthly_savings")
        s["tool_calls"] = sum(e["kind"] == "tool" for e in st["trace"])
        s["tool_errors"] = sum(e["kind"] == "tool" and not e["ok"] for e in st["trace"])
        rows.append(s)
    out = {"forecast": env.forecast_metrics, "tasks": rows}
    print(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    main()
