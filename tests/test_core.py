import pytest

from atlas import optimizer
from atlas.agents import PERMISSIONS
from atlas.bus import Bus
from atlas.supervisor import Supervisor, rule_plan
from atlas.tools import ToolBelt, ToolPermissionError, Trace, build_tools
from atlas.twin import Twin


def test_twin_cost_and_constraints():
    t = Twin.default()
    r = t.evaluate(1.0)
    assert r["active_servers"] == 20 and r["monthly_cost"] > 0


def test_consolidation_respects_constraints():
    t = Twin.default()
    new = optimizer.consolidate(t, None, 1.2)
    r = new.evaluate(1.2)
    assert r["safe"] and r["active_servers"] < 20
    for svc, hosts in new.placement.items():
        assert len(set(hosts)) == len(hosts)  # anti-affinity


def test_failure_reschedule_keeps_replicas():
    t = Twin.default(); t.servers["srv-03"].online = False
    t2 = optimizer.reschedule_after_failure(t, "srv-03", 1.0)
    assert all("srv-03" not in h for h in t2.placement.values())


def test_forecaster_beats_baseline(env):
    m = env.forecast_metrics
    assert m["gbm_mape"] < m["seasonal_naive_mape"]


def test_anomalies_find_injected_incidents(env):
    svcs = {a["service"] for a in build_tools(env)["detect_anomalies"]()}
    assert {"batch-etl", "recommender"} <= svcs


def test_rag_retrieves_right_incident(env):
    top = env.retriever.search("batch-etl cron schedule changed hourly overlapping runs", 1)[0]
    assert top["source"].startswith("INC-2025-031")


def test_guardrail_blocks_unpermitted_tool(env):
    belt = ToolBelt(build_tools(env), PERMISSIONS["forecasting"], "forecasting", Trace())
    with pytest.raises(ToolPermissionError):
        belt.call("apply_plan", placement={}, approved_by="x")


def test_rule_planner_what_if():
    p = rule_plan("What happens if server 12 goes offline and traffic doubles?")
    assert p["what_if"] == {"fail_server": "srv-12", "surge": 2.0}


def test_end_to_end_with_approval(env):
    sup = Supervisor(env)
    st = sup.run("Costs increased 30%. Investigate the cause and recommend a safe optimization plan.")
    assert st["status"] == "awaiting_approval"
    causes = " ".join(f["likely_cause"] for f in st["outputs"]["investigation"]["findings"])
    assert "cron" in causes
    before = env.twin.evaluate(env.demand_mult())["active_servers"]
    st = sup.approve(st["id"], "tester")
    assert st["status"] == "applied"
    assert env.twin.evaluate(env.demand_mult())["active_servers"] < before


def test_resume_from_checkpoint(env):
    sup = Supervisor(env)
    st = sup.run("Forecast demand for next week")
    st2 = sup.run("", resume_id=st["id"])
    assert st2["resumed"] and st2["done"] == st["done"]
