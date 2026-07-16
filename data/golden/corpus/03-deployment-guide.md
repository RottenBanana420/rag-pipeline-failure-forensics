# Deployment Guide

## Pipeline Overview

Every change to a Northwind service goes through four pipeline stages, run by GitHub Actions: **build**, **test**, **deploy-staging**, **deploy-prod**.

## Promotion Process

Promotion from staging to production requires no additional approval — staging deploys are automatic once tests pass. Promotion to **production**, however, requires a manual approval gate: either a Staff Engineer or the current on-call Incident Commander must approve the production deploy before it proceeds. See the On-Call Rotation Policy's "Who Is On Call" section for how to find out who currently holds either role.

## Rollback Procedure

If a deployment causes problems, run:

```
northwind-cli rollback <service> <previous-version>
```

Rollback must be initiated within 15 minutes of confirming a bad deploy, and the on-call engineer executing it should announce it in the incident channel first. This is the exact procedure referenced by the Incident Response Runbook's "Bad Deploy Scenario" section.

## Feature Flags

Risky changes should ship behind a flag in **Switchboard**, Northwind's internal feature-flagging tool, rather than as a direct code change. Flags let a change be disabled instantly without a rollback.

## Deployment Windows

No production deployments may be started after **2:00 PM Pacific on Fridays** — this gives the team a full business day to respond if something goes wrong before the weekend. In addition, there is a full deployment freeze from **December 15 through January 2** each year (the Q4 code freeze), during which no production deploys are permitted except emergency security patches approved by a Staff Engineer.
