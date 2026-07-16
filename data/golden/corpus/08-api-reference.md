# API Reference

## Authentication

Every request must include an `Authorization: Bearer <api_key>` header. API keys expire **90 days** after issuance and must be rotated before then.

## Rate Limits

- **Free tier**: up to 60 requests per minute.
- **Pro tier**: up to 600 requests per minute.

Requests beyond these limits receive a `429` error (see Error Codes below).

## Endpoints Overview

| Endpoint | Backing Service |
|---|---|
| `/v1/search` | Search Service |
| `/v1/ingest` | Ingestion Service |
| `/v1/notify` | Notification Service |

## Error Codes

| Code | Meaning |
|---|---|
| 400 | Invalid request |
| 401 | Unauthorized (missing or invalid API key) |
| 429 | Rate limit exceeded |
| 500 | Internal error |

## Versioning & Deprecation

When an API version is deprecated, Northwind announces the change and gives customers **60 days** of notice before the old version stops working.
