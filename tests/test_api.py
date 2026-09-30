from fastapi.testclient import TestClient

from atlas.api import app

c = TestClient(app)


def test_api_flow():
    assert c.get("/api/health").json()["ok"]
    assert c.get("/").status_code == 200
    s = c.get("/api/state").json(); assert len(s["servers"]) == 20
    assert len(c.get("/api/telemetry").json()["forecast"]["cores"]) == 168
    s = c.post("/api/inject", json={"fail_server": "srv-05"}).json()
    assert not next(v for v in s["servers"] if v["id"] == "srv-05")["online"]
    r = c.post("/api/ask", json={"query": "Optimize: consolidate onto fewer servers and reduce cost safely"}).json()
    assert r["status"] in ("awaiting_approval", "completed")
    if r["status"] == "awaiting_approval":
        assert c.post(f"/api/runs/{r['id']}/approve", json={"approver": "t"}).json()["status"] == "applied"
