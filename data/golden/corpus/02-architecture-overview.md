# System Architecture Overview

This document describes Northwind's service architecture at a high level.

## Service Inventory

- **API Gateway** — written in Go. Routes all external client traffic.
- **Ingestion Service** — written in Python. Accepts document uploads and writes them to blob storage.
- **Search Service** — written in Python. Handles query requests against the indexed corpus.
- **Notification Service** — written in Node.js. Publishes events (e.g. ingestion complete, search alerts) to subscribers.

## Data Flow

A client request enters through the API Gateway, which routes search queries to the Search Service and document uploads to the Ingestion Service. After the Ingestion Service finishes processing an upload, it emits an event that the Notification Service delivers to subscribed clients.

## Datastores

- **Postgres** — the primary metadata store (documents, users, permissions). Owned by the Platform Team.
- **Redis** — the caching layer in front of Postgres and the Search Service. Owned by the Platform Team.
- **S3** — blob storage for raw uploaded documents. Owned by the Data Team.

## Environments

Northwind runs three environments: `dev`, `staging`, and `prod`. Code moves through these environments in order; see the Deployment Guide for exactly how promotion between them works.

## Service Ownership Table

| Service | Owning Team |
|---|---|
| API Gateway | Platform Team |
| Ingestion Service | Data Team |
| Search Service | Data Team |
| Notification Service | Product Team |
