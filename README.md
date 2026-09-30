# ATLAS — Autonomous AI Digital Twin

A multi-agent system that monitors a simulated data center, forecasts demand, investigates incidents with RAG,
proposes cost optimizations, **tests every proposal in a digital twin**, verifies it independently for safety,
and waits for **human approval** before applying anything.

> Agents choose tools and interpret results. Numbers (forecasts, constraints, simulations) always come from
> deterministic code and trained models, never from the LLM.

## Quick start (5 minutes)

Requires Python 3.10+.

```bash
cd atlas
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

pytest -q                          # 11 tests should pass
uvicorn atlas.api:app --port 8000  # open http://localhost:8000
```

Startup takes ~5 s (it generates 60 days of telemetry and trains the models).

**Optional — use Claude for planning and executive summaries:**
```bash
export ANTHROPIC_API_KEY=sk-ant-...        # Windows PowerShell: $env:ANTHROPIC_API_KEY="sk-ant-..."
```
Without a key everything still works with the offline rule-based planner. With a key, the supervisor plans with a
fast model and writes the report with a stronger one (model routing, see `atlas/llm.py`). If Claude fails, ATLAS falls
back to the rules automatically.

### Other ways to run
| Command | What it does |
|---|---|
| `python -m atlas.cli` | Runs the flagship "costs up 30%" investigation in the terminal |
| `python -m atlas.cli "What happens if server 9 goes offline and traffic +30%?"` | Any question |
| `python -m atlas.cli --approve` | Runs and applies the approved plan to the twin |
| `python -m atlas.evaluate` | Evaluation harness (diagnosis recall, grounding, plan accuracy, safety, forecast MAPE) |
| `python -m atlas.mcp_server` | Exposes the tools over MCP (stdio) for Claude Desktop / Claude Code |
| `docker compose up --build` | Containerised run on port 8000 |

## Demos (buttons in the dashboard)
1. **Incident investigation** — finds the two injected causes (batch-etl cron change, recommender memory leak) from anomalies + logs + past incidents.
2. **Forecast** — 7-day cluster CPU forecast; gradient boosting vs seasonal-naive baseline on a held-out week.
3. **What-if** — server failure + traffic surge simulated in the twin; shows which servers break limits.
4. **Optimization** — candidate consolidations; the Safety agent rejects any that fail peak CPU, memory, anti-affinity, rack spread, N+1 or a +20% surge.
5. **Human approval** — click *Approve & apply*; the twin updates and the change is audited. Click a server tile to inject a failure first.

## Architecture
```
Dashboard / CLI / MCP client
        │
   Supervisor  ── plan (Claude or rules) → parallel steps → reflect/replan → decide → await approval
        │           checkpoints every step to runs/<id>.json (resume: POST /api/runs/{id}/resume)
        │           long-term memory: runs/memory.json
 ┌──────┼───────────┬──────────────┬─────────────┬────────────┬─────────┐
Monitoring  Forecasting  Investigation  Optimization  Simulation  Safety    Executor (approval only)
(anomalies) (GBM model)  (logs + RAG)   (candidates)  (twin)      (independent re-check)
        └──── A2A-style message bus (bus.py) ─── permission-scoped ToolBelt + trace (tools.py)
                                  │
      Digital twin (twin.py) · optimizer (optimizer.py) · ML (ml.py) · hybrid RAG (rag.py) · data (data.py)
```

| File | Role |
|---|---|
| `atlas/twin.py` | Servers, services, placement, cost, constraint checks, N+1 test |
| `atlas/optimizer.py` | Constraint-aware bin packing (first-fit decreasing) + failure rescheduling |
| `atlas/data.py` | Synthetic telemetry with injected incidents + logs |
| `atlas/ml.py` | HistGradientBoosting forecaster, IsolationForest anomaly detection |
| `atlas/rag.py` | BM25 + TF-IDF hybrid retrieval, reciprocal rank fusion, rerank |
| `atlas/tools.py` | Tool registry, per-agent permissions (guardrails), tracing |
| `atlas/agents.py` | Seven agents and their tool permissions |
| `atlas/supervisor.py` | Planning, parallel execution, retries, reflection/replanning, checkpoints, memory, approval |
| `atlas/api.py` + `static/index.html` | FastAPI backend and dashboard |
| `atlas/mcp_server.py` | MCP server (apply_plan deliberately not exposed) |
| `atlas/evaluate.py` | Evaluation harness |
| `knowledge/` | Runbooks and past incidents used by RAG — add your own `.md` files |

## API
`GET /api/state` · `GET /api/telemetry` · `POST /api/ask {"query"}` · `GET /api/runs/{id}` ·
`POST /api/runs/{id}/approve|reject {"approver"}` · `POST /api/runs/{id}/resume` ·
`POST /api/inject {"fail_server":"srv-05"}` · `POST /api/reset`. Interactive docs at `/docs`.

## Use with Claude Desktop (MCP)
Add to `claude_desktop_config.json`:
```json
{"mcpServers": {"atlas": {"command": "python", "args": ["-m", "atlas.mcp_server"], "cwd": "/path/to/atlas"}}}
```

## Measured results (offline, seed-fixed)
- Forecast, 7-day holdout: GBM MAPE **2.7%** vs seasonal-naive **6.2%**
- Diagnosis recall on injected incidents: **2/2**, each backed by logs + a matching past incident
- Recommended plan: 20 → 14 servers, **$11,972 → $8,380/month**, peak CPU 82%, passes N+1 and +20% surge
- Current placement is flagged unsafe (two servers >85% peak) — fixing the ETL cron alone makes it safe

These come from synthetic data; state that clearly in interviews and on your resume.

## Upgrade path (to match the full spec)
| Now | Upgrade |
|---|---|
| Custom supervisor loop | LangGraph `StateGraph` with a Postgres checkpointer (same nodes) |
| First-fit-decreasing packing | OR-Tools CP-SAT for provably minimal servers |
| Static twin evaluation | SimPy discrete-event sim for queueing / latency |
| TF-IDF "dense" retrieval | Real embeddings in PostgreSQL + pgvector; cross-encoder rerank |
| GBM forecaster | Add LSTM / Temporal Fusion Transformer and compare |
| Synthetic generator | Your conditional WGAN-GP for realistic workloads and rare failures |
| In-process bus | Real A2A agent cards/endpoints; Redis queue |
| HTML dashboard | Next.js front end; deploy to GCP Cloud Run |
