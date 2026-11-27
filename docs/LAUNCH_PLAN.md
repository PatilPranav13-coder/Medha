# MedhaAI public beta launch plan

Target date: 1 January 2027. The launch is a public beta for students and engineers researching AI, space/ISRO, and engineering topics. The model-selection decision is a locally served open-weights instruct model; the owner will choose its name and quantisation after running the capability harness on the target hardware.

## Build order

1. **WP1 — Plan and architecture:** make `backend/` the app entry point, serve the Medha OS client there, document decisions, and preserve the from-scratch model as an experimental adapter.
2. **WP2 — Brain adapter:** add a common streaming interface for the experimental model, OpenAI-compatible local servers, and a deterministic test stub. Bound context, output, retries, queueing, and timeouts. Do not choose a production model on the owner's behalf.
3. **WP3 — Accounts and persistence:** add SQLite migrations, invite-only signup, scrypt password storage, hashed sessions, CSRF checks, lockout, and per-user ownership checks.
4. **WP4 — Chat product:** wire persisted conversations to streamed SSE responses and replace simulated client behavior with API-backed sign-in and chat.
5. **WP5 — Search:** add provider fallback, safe page fetching, ranking, cited answers, and deterministic fixture tests.
6. **WP6 — Capability harness:** create original tasks and executable checkers. The real-model capability gate stays pending until the owner runs it on the chosen hardware.
7. **WP7 — Safety:** implement abuse controls, crisis and severe-harm behavior, and red-team tests.
8. **WP8 — Privacy and legal drafts:** implement consent, export, deletion, retention, and draft policy documents for owner/legal review.
9. **WP9 — Operations:** add deploy packaging, health checks, backups, flags, CI and operations documentation.
10. **P1 — Memory/files, admin/feedback, and user docs** after the P0 work is stable. P2 features are cut first if the schedule slips.

## Milestone stops

- 4 Dec: WP1–WP4. Stop for owner review.
- 11 Dec: WP5–WP7. Stop for owner review and model-harness run.
- 18 Dec: WP8–WP9 and P1 items that fit. Freeze features after this date.
- 25 Dec: run and report gates, docs, and staging verification. Stop for owner review.
- 26–31 Dec: fixes only, then waitlist soft launch and owner go/no-go.

The current work session targets milestone 1 only. No parity or production readiness claim is made until the later gates have actually run.
