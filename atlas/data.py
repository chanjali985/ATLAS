"""Synthetic telemetry + logs for the twin (swap for real Prometheus data or a WGAN-GP generator)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .twin import Twin

DAYS = 60
INCIDENT_START_H = 24 * 45  # day 45: batch-etl misconfigured cron -> runs hourly


def generate_telemetry(t: Twin, days: int = DAYS, seed: int = 42) -> pd.DataFrame:
    """Hourly per-service CPU demand (cores) and per-server metrics."""
    rng = np.random.default_rng(seed)
    hours = pd.date_range("2026-07-01", periods=days * 24, freq="h")
    h = np.arange(len(hours))
    daily = 1 + 0.35 * np.sin(2 * np.pi * (h % 24 - 8) / 24)
    weekly = np.where(hours.dayofweek >= 5, 0.8, 1.0)
    trend = 1 + 0.004 * h / 24  # ~0.4%/day organic growth
    rows = []
    for svc in t.services.values():
        demand = svc.base_cpu * daily * weekly * trend * rng.normal(1, 0.05, len(h))
        if svc.id == "batch-etl":
            spike = (h >= INCIDENT_START_H) * svc.base_cpu * 2.2
            demand = demand + spike
        if svc.id == "recommender":
            leak = np.clip(h - INCIDENT_START_H - 72, 0, None) * 0.03  # slow memory leak -> CPU (GC)
            demand = demand + leak
        rows.append(pd.DataFrame({"ts": hours, "service": svc.id, "cpu_cores": demand}))
    df = pd.concat(rows, ignore_index=True)
    return df


def server_metrics(t: Twin, svc_df: pd.DataFrame) -> pd.DataFrame:
    """Project service demand onto servers using current placement."""
    rows = []
    for svc_id, hosts in t.placement.items():
        part = svc_df[svc_df.service == svc_id]
        for hst in hosts:
            rows.append(pd.DataFrame({"ts": part.ts.values, "server": hst,
                                      "cpu": part.cpu_cores.values / len(hosts),
                                      "mem": t.services[svc_id].base_mem / 2}))
    m = pd.concat(rows).groupby(["ts", "server"], as_index=False).sum()
    m["cpu_util"] = m.apply(lambda r: r.cpu / t.servers[r.server].cpu_cores, axis=1)
    m["power_w"] = 120 + 280 * m["cpu_util"]
    return m


def generate_logs(svc_df: pd.DataFrame) -> list[dict]:
    t0 = svc_df.ts.min() + pd.Timedelta(hours=INCIDENT_START_H)
    return [
        {"ts": str(t0 - pd.Timedelta(hours=1)), "service": "batch-etl", "level": "INFO",
         "msg": "Deploy v2.14.0: cron schedule changed '0 2 * * *' -> '0 * * * *' (PR #812)"},
        {"ts": str(t0), "service": "batch-etl", "level": "WARN",
         "msg": "Job etl-daily-rollup started while previous run still active; full table rescan"},
        {"ts": str(t0 + pd.Timedelta(hours=5)), "service": "batch-etl", "level": "WARN",
         "msg": "Autoscaler added 4 workers: queue depth 1830"},
        {"ts": str(t0 + pd.Timedelta(hours=80)), "service": "recommender", "level": "WARN",
         "msg": "GC pause 1.8s; heap usage 91% (model cache not evicting)"},
        {"ts": str(t0 + pd.Timedelta(hours=200)), "service": "recommender", "level": "ERROR",
         "msg": "OOMKilled pod recommender-7f9c; restarted"},
        {"ts": str(t0 + pd.Timedelta(hours=30)), "service": "checkout", "level": "INFO",
         "msg": "p99 latency 210ms (normal)"},
    ]
