# New Engineer Onboarding Guide

Welcome to Northwind! This guide walks new engineers through their first two weeks.

## Welcome & Team Structure

Northwind's engineering organization is split into three teams: the **Platform Team** (infrastructure, API Gateway, core datastores), the **Data Team** (ingestion and search), and the **Product Team** (customer-facing features and notifications). Every new engineer is assigned to exactly one team before their start date; your recruiter will tell you which one.

## Local Development Setup

1. Clone the monorepo and install the `northwind-cli` tool.
2. Run `northwind-cli bootstrap` to pull down local copies of the services you'll work on.
3. For a full list of Northwind's services and which team owns each one, see the **Service Ownership Table** in the Architecture Overview document — you'll need it to know who to ask when something breaks.

## Access & Permissions

New engineers must request cloud console access, source control access, and internal tool access through the `#it-help` Slack channel during their first day. Access isn't granted permanently and without review — see the Security Policy's Access Control Principles section for how often these grants are re-checked.

## First-Week Checklist

1. Complete mandatory security training (day 1).
2. Get your local environment running and pass the onboarding smoke test (by day 3).
3. Shadow one on-call shift with your team's on-call engineer (during week 1).
4. Ship your first change to staging (by day 5).

## Where to Get Help

Post questions in `#eng-help`. If your question is about who's currently on call or how the on-call rotation works, check the On-Call Rotation Policy document instead — `#eng-help` is for general engineering questions, not paging.
