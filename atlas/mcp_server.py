"""Expose ATLAS tools over MCP (stdio) so any MCP client (Claude Desktop, Claude Code) can drive the twin.

Run:  python -m atlas.mcp_server
"""
from mcp.server.fastmcp import FastMCP

from .tools import Environment, build_tools

env = Environment()
T = build_tools(env)
mcp = FastMCP("atlas-digital-twin")


@mcp.tool()
def get_metrics(window_h: int = 168) -> dict:
    """Cluster CPU/power metrics and top-growing services over the window."""
    return T["get_metrics"](window_h=window_h)


@mcp.tool()
def detect_anomalies() -> list[dict]:
    """Services with anomalous demand in the last 14 days (IsolationForest)."""
    return T["detect_anomalies"]()


@mcp.tool()
def forecast_demand(horizon_h: int = 168) -> dict:
    """Forecast total cluster CPU demand (cores) for the horizon."""
    return T["forecast_demand"](horizon_h=horizon_h)


@mcp.tool()
def search_logs(service: str = "", level: str = "") -> list[dict]:
    """Search operational logs by service and level."""
    return T["search_logs"](service=service or None, level=level or None)


@mcp.tool()
def search_knowledge(query: str, k: int = 3) -> list[dict]:
    """Hybrid (BM25 + TF-IDF) search over runbooks and past incidents."""
    return T["search_knowledge"](query=query, k=k)


@mcp.tool()
def simulate(consolidate_to: int = 0, fail_server: str = "", demand_mult: float = 0.0, fix_etl: bool = False) -> dict:
    """Run a what-if scenario in the digital twin (read-only; never changes real state)."""
    sc = {"consolidate_to": consolidate_to or None, "fail_server": fail_server or None,
          "demand_mult": demand_mult or None, "fix_etl": fix_etl}
    r = T["simulate"](scenario=sc)
    r.pop("placement", None)
    return r

# NOTE: apply_plan is deliberately NOT exposed over MCP — changes require human approval via the API.

if __name__ == "__main__":
    mcp.run()
