"""Digital twin of a small data center: servers, services, placement, demand, cost.

Everything here is deterministic, plain Python. Agents call into it as a tool;
they never replace it.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd

HOURS_PER_MONTH = 730


@dataclass
class Server:
    id: str
    cpu_cores: float = 16
    mem_gb: float = 64
    cost_per_hour: float = 0.82  # ~ $600/month
    rack: str = "r1"
    online: bool = True


@dataclass
class Service:
    id: str
    base_cpu: float          # cores at average demand
    base_mem: float          # GB (roughly flat)
    peak_factor: float = 1.6 # peak / average
    replicas: int = 2        # min replicas needed for availability
    critical: bool = False


@dataclass
class Twin:
    servers: dict[str, Server] = field(default_factory=dict)
    services: dict[str, Service] = field(default_factory=dict)
    # placement[service] -> list of server ids (one per replica)
    placement: dict[str, list[str]] = field(default_factory=dict)
    cpu_limit: float = 0.85  # max safe CPU utilization at peak
    mem_limit: float = 0.90

    # ---------- construction ----------
    @classmethod
    def default(cls, n_servers: int = 20, seed: int = 7) -> "Twin":
        rng = np.random.default_rng(seed)
        t = cls()
        for i in range(n_servers):
            sid = f"srv-{i+1:02d}"
            t.servers[sid] = Server(sid, rack=f"r{i // 5 + 1}")
        names = ["api-gateway", "auth", "checkout", "search", "recommender",
                 "catalog", "payments", "inventory", "notifications", "analytics",
                 "ml-inference", "reporting", "media", "user-profile", "batch-etl"]
        for n in names:
            crit = n in {"api-gateway", "auth", "checkout", "payments"}
            t.services[n] = Service(
                n,
                base_cpu=float(rng.uniform(4, 9)),
                base_mem=float(rng.uniform(10, 30)),
                peak_factor=float(rng.uniform(1.3, 1.8)),
                replicas=3 if crit else 2,
                critical=crit,
            )
        t.spread_placement()
        return t

    def clone(self) -> "Twin":
        return copy.deepcopy(self)

    def spread_placement(self) -> None:
        """Naive round-robin placement (how the 'current' infra looks)."""
        online = [s for s in self.servers if self.servers[s].online]
        # interleave racks (r1,r2,r3,r4,r1,...) so consecutive replicas land on different racks
        online.sort(key=lambda s: (int(s.split("-")[1]) - 1) % 5)
        i = 0
        self.placement = {}
        for svc in self.services.values():
            self.placement[svc.id] = []
            for _ in range(svc.replicas):
                self.placement[svc.id].append(online[i % len(online)])
                i += 1

    # ---------- evaluation ----------
    def load(self, demand_mult: float = 1.0, peak: bool = True) -> pd.DataFrame:
        rows = {s: {"cpu": 0.0, "mem": 0.0} for s in self.servers}
        for svc_id, hosts in self.placement.items():
            svc = self.services[svc_id]
            if not hosts:
                continue
            f = svc.peak_factor if peak else 1.0
            per_cpu = svc.base_cpu * f * demand_mult / len(hosts)
            for h in hosts:
                rows[h]["cpu"] += per_cpu
                rows[h]["mem"] += svc.base_mem / 2  # each replica holds ~half the service footprint
        df = pd.DataFrame(rows).T
        df["cpu_cap"] = [self.servers[s].cpu_cores for s in df.index]
        df["mem_cap"] = [self.servers[s].mem_gb for s in df.index]
        df["online"] = [self.servers[s].online for s in df.index]
        df["cpu_util"] = df["cpu"] / df["cpu_cap"]
        df["mem_util"] = df["mem"] / df["mem_cap"]
        df["active"] = (df["cpu"] > 0) & df["online"]
        return df

    def evaluate(self, demand_mult: float = 1.0) -> dict:
        peak = self.load(demand_mult, peak=True)
        avg = self.load(demand_mult, peak=False)
        active = peak[peak["active"]]
        violations = []
        for sid, r in peak.iterrows():
            if r["cpu"] > 0 and not r["online"]:
                violations.append(f"{sid} is offline but hosts workloads")
            if r["cpu_util"] > self.cpu_limit:
                violations.append(f"{sid} peak CPU {r['cpu_util']:.0%} > {self.cpu_limit:.0%}")
            if r["mem_util"] > self.mem_limit:
                violations.append(f"{sid} memory {r['mem_util']:.0%} > {self.mem_limit:.0%}")
        for svc_id, hosts in self.placement.items():
            svc = self.services[svc_id]
            live = [h for h in hosts if self.servers[h].online]
            if len(set(live)) < svc.replicas:
                violations.append(f"{svc_id} has {len(set(live))}/{svc.replicas} healthy replicas")
            if svc.critical and len({self.servers[h].rack for h in live}) < 2:
                violations.append(f"{svc_id} (critical) not spread across 2+ racks")
        n_active = int(active.shape[0])
        cost = sum(self.servers[s].cost_per_hour for s in active.index) * HOURS_PER_MONTH
        return {
            "active_servers": n_active,
            "avg_cpu_util": float(avg.loc[active.index, "cpu_util"].mean()) if n_active else 0.0,
            "peak_cpu_util_max": float(active["cpu_util"].max()) if n_active else 0.0,
            "monthly_cost": round(cost, 2),
            "violations": violations,
            "safe": not violations,
        }

    def n_plus_one_ok(self, demand_mult: float = 1.0) -> tuple[bool, str | None]:
        """Can we lose any single active server and still be safe (after reschedule)?"""
        from .optimizer import reschedule_after_failure
        active = [s for s, r in self.load(demand_mult).iterrows() if r["active"]]
        for s in active:
            t = self.clone()
            t.servers[s].online = False
            t = reschedule_after_failure(t, s, demand_mult)
            if not t.evaluate(demand_mult)["safe"]:
                return False, s
        return True, None

    def to_dict(self) -> dict:
        return {
            "servers": {k: asdict(v) for k, v in self.servers.items()},
            "services": {k: asdict(v) for k, v in self.services.items()},
            "placement": self.placement,
        }
