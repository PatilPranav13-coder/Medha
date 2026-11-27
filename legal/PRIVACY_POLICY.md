# DRAFT — not legal advice, needs review

**Last updated:** 27 November 2026  
**Contact:** `GRIEVANCE_EMAIL` (the operator must configure a monitored address before launch).

## What Medha stores

Medha stores your email address, a salted password verifier, session and consent records, conversations and assistant responses, source links, feedback, and any memories or files you choose to save. The product also records limited operational events such as signup and feature counts. These events must not contain message text, passwords, session tokens, or uploaded-file contents.

## Why it is used

The data is used to authenticate you, provide chat and research features, retrieve your saved memories and files, keep the service reliable, prevent abuse, and respond to feedback. The beta has no third-party advertising tracker.

## Models and search providers

The beta is designed to use a locally served model selected by the operator. Chat content is not sent to a hosted model API by default. When web search is selected, Medha sends the query to the configured Brave Search API, SearXNG instance, or Wikipedia service. Search providers and linked websites have their own privacy practices. Do not include information in a search request that you do not want sent to that provider.

## Storage and retention

Account data is stored in the operator's SQLite database and uploaded files are stored outside the public web directory. Operational event records are retained for 30 days by default; an operator may configure a different period. Backups may retain deleted data until their rotation period expires. The operator must set and publish a backup-retention period before launch.

## Your controls

You can export your data as JSON, delete all conversations, manage saved memories and files, or delete your account. Account deletion removes its database records and uploaded files. Contact `GRIEVANCE_EMAIL` if a product control does not work.

## Security and incidents

Medha uses invite-only signup, password hashing, secure session cookies, request limits, and access controls. No internet-connected service can promise perfect security. The operator follows `docs/INCIDENT_RESPONSE.md` and will confirm any legally required notice and deadlines with counsel before launch.

## Children and sensitive uses

The beta is intended for students and engineers and is not designed for children or for making decisions about a person. Do not submit highly sensitive personal information. The operator must obtain legal review of age and sensitive-data rules before opening signup.

## Legal status

This is a draft product description, not legal advice. It does not claim compliance with any law. The owner and a qualified lawyer must review it, confirm the grievance contact, and verify the applicable DPDP commencement dates before launch.
