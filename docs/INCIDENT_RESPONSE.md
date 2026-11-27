# Incident response runbook

## Preparation

- Keep one operator on call and maintain access to the host, SQLite volume, reverse proxy, and model endpoint.
- Set `MEDHA_MAINTENANCE=true` to stop new writes and chat traffic. If the host is compromised, isolate it at the firewall before investigating.
- Preserve request IDs and counts-only operational events. Do not copy message bodies, passwords, session tokens, or uploaded-file contents into tickets or logs.
- Keep an encrypted offsite backup and document who can restore it. Backups may contain deleted data until rotation expires.

## Triage and containment

1. Record the detection time, affected service, operator, and request IDs. Avoid collecting unnecessary personal data.
2. Enable maintenance mode or stop the affected service. Restrict network access if an attacker may still have access.
3. Rotate exposed model, search, database, and host credentials. Revoke affected sessions if account access may be compromised.
4. Preserve a read-only copy of relevant system logs and the database for investigation, with access limited to the incident responders.
5. Determine which systems and categories of data were affected, the likely period, and whether the incident is still active.

## Recovery

1. Patch the cause and verify the fix in staging.
2. Restore from a known-good backup only after checking its integrity and the target path.
3. Re-enable traffic gradually; check `/healthz`, `/readyz`, error rates, and the admin counters endpoint.
4. Record decisions, recovery time, and follow-up work without reproducing user content.

## Notifications and legal review

**Owner to confirm with a lawyer:** whether the incident triggers a notice, who must receive it, what the notice must contain, and every applicable notification deadline. Do not rely on this runbook as a legal determination. Record counsel's advice and the relevant rule version in the incident record.

The Digital Personal Data Protection Act, 2023 and the Digital Personal Data Protection Rules, 2025 have staggered commencement dates. Date arithmetic from the official 13 November 2025 notifications gives **13 November 2026** for the one-year phase (including Rule 4 and the cited Act provisions) and **13 May 2027** for the eighteen-month phase (including Rules 3, 5–16, 22 and 23 and the cited Act provisions). **Owner to confirm each provision and effective date with a lawyer before launch.** Sources: [Act commencement notification](https://www.meity.gov.in/static/uploads/2025/11/c56ceae6c383460ca69577428d36828b.pdf) and [DPDP Rules, 2025 notification](https://www.meity.gov.in/static/uploads/2025/11/53450e6e5dc0bfa85ebd78686cadad39.pdf).
