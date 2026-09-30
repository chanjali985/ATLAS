# Runbook: Server consolidation
Before consolidating workloads onto fewer servers:
1. Peak CPU per server must stay under 85% and memory under 90%.
2. Every service keeps its minimum replica count on distinct hosts (anti-affinity).
3. Critical services (api-gateway, auth, checkout, payments) must span at least two racks.
4. The cluster must survive the loss of any single server (N+1).
5. Changes to production require human approval and a rollback plan; migrate one rack at a time.
