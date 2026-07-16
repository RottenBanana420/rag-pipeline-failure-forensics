# Incident Response Runbook

## Severity Levels

| Severity | Definition | Response Target |
|---|---|---|
| SEV1 | Full outage, all customers affected | Acknowledge within 15 minutes |
| SEV2 | Partial outage or major degradation | Acknowledge within 30 minutes |
| SEV3 | Minor degradation, limited customer impact | Acknowledge within 4 hours |
| SEV4 | Cosmetic issue, no customer impact | Next business day |

## Declaring an Incident

Post in the `#incidents` Slack channel and page the on-call engineer using **Signal**, Northwind's paging tool. Include the affected service and your best guess at severity — it can be revised later.

## Escalation Path

For SEV1 and SEV2 incidents specifically, Signal will automatically page the secondary on-call engineer if the primary does not acknowledge the page within **10 minutes**.

## Bad Deploy Scenario

If an incident is traced to a recent deployment: (1) confirm the correlation using the error-rate dashboard, (2) execute the rollback procedure described in the Deployment Guide's "Rollback Procedure" section, (3) verify error rates return to baseline and hold there for at least 15 minutes before closing the incident.

## Postmortem Requirements

A written postmortem is required for every SEV1 or SEV2 incident, and is due within **3 business days** of the incident being resolved.
