"""Command-line entry: python -m atlas.cli "your question" [--approve]"""
import argparse
import json

from .supervisor import Supervisor
from .tools import Environment


def main():
    p = argparse.ArgumentParser()
    p.add_argument("query", nargs="?", default="Our cloud costs increased by 30% this month. Investigate the cause, "
                   "forecast next week's demand, simulate cost-saving strategies and recommend a safe plan.")
    p.add_argument("--approve", action="store_true", help="approve and apply the recommendation to the twin")
    p.add_argument("--json", action="store_true")
    a = p.parse_args()
    print("Building digital twin, telemetry and models...")
    sup = Supervisor(Environment())
    st = sup.run(a.query)
    if a.json:
        print(json.dumps(st, default=str, indent=1)); return
    print("\nPLAN:", " -> ".join(st["plan"]["steps"]), st["plan"].get("what_if") or "")
    print(st["report"])
    for r in st.get("decision", {}).get("comparison", []):
        print(f"  {r['name']:<26} servers={r['active_servers']!s:<4} cost={r['monthly_cost']!s:<9} "
              f"peak={r['peak_cpu_util_max'] and round(r['peak_cpu_util_max'],2)!s:<5} "
              f"verdict={'OK' if r.get('approved') or (r.get('approved') is None and r.get('safe')) else 'NO'} "
              f"{'; '.join(r['reasons'][:2])}")
    print(f"\nstatus: {st['status']}   run id: {st['id']}")
    if a.approve and st["status"] == "awaiting_approval":
        st = sup.approve(st["id"], "cli-user")
        print("APPLIED:", st["execution"]["result"])


if __name__ == "__main__":
    main()
