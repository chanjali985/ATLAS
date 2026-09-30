# INC-2025-031: Cost spike from batch ETL schedule change
Summary: A deploy changed the batch-etl cron from daily to hourly. Overlapping runs rescanned full tables, the autoscaler added workers and cluster CPU rose ~25%, raising monthly compute cost.
Root cause: cron schedule misconfiguration in batch-etl deploy.
Fix: revert cron to '0 2 * * *', add a job lock to prevent overlapping runs, add CI lint for cron expressions.
Outcome: CPU returned to baseline within 2 hours; cost normalized next billing cycle.
