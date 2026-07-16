# Security Policy

## Access Control Principles

Northwind follows the principle of least privilege: engineers are granted only the access they need for their current role. All access grants are reviewed **quarterly**, and unused permissions are revoked automatically.

## Credential Management

All employees must store credentials in **1Password** and enable multi-factor authentication through **Okta** for every internal system.

## Incident Reporting

Any suspected security incident must be reported immediately in `#security-incidents` and treated as **at minimum a SEV2** under the Incident Response Runbook's severity levels, even if customer impact isn't yet confirmed. Because security incidents carry additional legal and disclosure obligations, their postmortems are due within **5 business days** of resolution.

## Data Classification

Northwind classifies all data into four tiers:

- **Public** — marketing materials, public documentation.
- **Internal** — internal docs, non-sensitive engineering metrics.
- **Confidential** — source code, internal financial data.
- **Restricted** — customer PII and payment data.

## Third-Party Vendor Review

Any new SaaS vendor that will handle Confidential or Restricted data must be approved by the Security team before purchase. See the Data Retention Policy's retention rules for how vendor-held data must be handled once a vendor is approved.
