"""Constraint-aware placement optimizer (first-fit-decreasing bin packing).

Goal: pack service replicas onto the fewest servers while respecting peak CPU,
memory, anti-affinity (replicas on different hosts) and rack spread for
critical services. Swap in OR-Tools CP-SAT later for provable optimality.
"""
from __future__ import annotations

from .twin import Twin


def _pack(t: Twin, allowed: list[str], demand_mult: float, headroom: float) -> dict | None:
    cpu = {s: 0.0 for s in allowed}
    mem = {s: 0.0 for s in allowed}
    cap_cpu = {s: t.servers[s].cpu_cores * t.cpu_limit * headroom for s in allowed}
    cap_mem = {s: t.servers[s].mem_gb * t.mem_limit for s in allowed}
    placement: dict[str, list[str]] = {}
    svcs = sorted(t.services.values(), key=lambda v: v.base_cpu * v.peak_factor, reverse=True)
    for svc in svcs:
        need_cpu = svc.base_cpu * svc.peak_factor * demand_mult / svc.replicas
        need_mem = svc.base_mem / 2
        hosts: list[str] = []
        racks: set[str] = set()
        # prefer fuller servers (tighter packing), but force rack spread for critical
        for _ in range(svc.replicas):
            order = sorted(allowed, key=lambda s: -cpu[s])
            if svc.critical and len(racks) < 2 and hosts:
                order = [s for s in order if t.servers[s].rack not in racks] + \
                        [s for s in order if t.servers[s].rack in racks]
            chosen = next((s for s in order if s not in hosts
                           and cpu[s] + need_cpu <= cap_cpu[s]
                           and mem[s] + need_mem <= cap_mem[s]), None)
            if chosen is None:
                return None
            hosts.append(chosen)
            racks.add(t.servers[chosen].rack)
            cpu[chosen] += need_cpu
            mem[chosen] += need_mem
        placement[svc.id] = hosts
    return placement


def consolidate(t: Twin, target_servers: int | None = None, demand_mult: float = 1.0,
                headroom: float = 1.0) -> Twin | None:
    """Return a new twin packed onto `target_servers` (or the minimum feasible)."""
    online = [s for s in sorted(t.servers) if t.servers[s].online]
    # interleave racks so the first N servers span racks
    by_rack: dict[str, list[str]] = {}
    for s in online:
        by_rack.setdefault(t.servers[s].rack, []).append(s)
    ordered = []
    while any(by_rack.values()):
        for r in sorted(by_rack):
            if by_rack[r]:
                ordered.append(by_rack[r].pop(0))
    counts = [target_servers] if target_servers else range(3, len(ordered) + 1)
    for n in counts:
        p = _pack(t, ordered[:n], demand_mult, headroom)
        if p:
            new = t.clone()
            new.placement = p
            return new
    return None


def reschedule_after_failure(t: Twin, failed: str, demand_mult: float = 1.0) -> Twin:
    """Move replicas off a failed server onto the least-loaded eligible servers."""
    t = t.clone()
    load = t.load(demand_mult)
    for svc_id, hosts in t.placement.items():
        if failed not in hosts:
            continue
        svc = t.services[svc_id]
        need = svc.base_cpu * svc.peak_factor * demand_mult / svc.replicas
        others = [h for h in hosts if h != failed]
        cands = sorted(
            (s for s in t.servers if t.servers[s].online and s not in others),
            key=lambda s: load.loc[s, "cpu_util"] if load.loc[s, "active"] else 9)
        # prefer already-active servers with room, else wake a spare
        pick = next((s for s in cands if load.loc[s, "active"] and
                     (load.loc[s, "cpu"] + need) / load.loc[s, "cpu_cap"] <= t.cpu_limit), None)
        pick = pick or next((s for s in cands if not load.loc[s, "active"]), None) or cands[0]
        t.placement[svc_id] = others + [pick]
        load.loc[pick, "cpu"] += need
        load.loc[pick, "active"] = True
    return t
