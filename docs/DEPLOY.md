# Deploy Medha on an Ubuntu host

This is an operator procedure for a single-host beta. It uses one Uvicorn worker because SQLite write coordination and the IP/user rate limiter are local to the process. It does not provision a server, domain, DNS, Sentry project, search credentials, or model weights.

## Prerequisites

- An Ubuntu VPS or GPU host with Docker Engine and the Compose plugin installed.
- A domain whose DNS A record points to the host. Add an AAAA record only when IPv6 is routed to the host.
- Inbound TCP ports 80 and 443, plus UDP 443 for HTTP/3, allowed at the host firewall and provider firewall.
- A model selected and installed by the owner. `MEDHA_BRAIN_MODEL` is deliberately blank in the example; startup fails until it names an available model.

## Configure the host

1. Clone the reviewed `launch/2027-01` revision on the host and copy `.env.example` to `.env`.
2. Set `MEDHA_DOMAIN`, `MEDHA_BRAIN_PROVIDER`, `MEDHA_BRAIN_BASE_URL`, and `MEDHA_BRAIN_MODEL`. Keep `MEDHA_COOKIE_SECURE=true`. Set a monitored `GRIEVANCE_EMAIL`, a strong one-use `MEDHA_BOOTSTRAP_INVITE_CODE`, and the initial user cap. Keep `.env` readable only by the operator.
3. Choose one brain route:
   - **Ollama profile:** set a model tag; start Ollama first with `docker compose --profile local-brain up -d ollama`, pull the owner-selected tag with `docker compose exec ollama ollama pull <model-tag>`, then start the app and Caddy with `docker compose up -d --build`.
   - **Another OpenAI-compatible server:** point `MEDHA_BRAIN_BASE_URL` at its private HTTPS or loopback-reachable endpoint, set its exact model tag, and omit the Ollama profile.
4. Configure Brave or SearXNG only if desired. Wikipedia remains the last no-key fallback. Search requests are sent to the provider selected by `MEDHA_SEARCH_PROVIDER`.
5. Start and inspect the service:

   ```sh
   docker compose up -d --build
   docker compose ps
   docker compose logs --tail=100 app caddy
   curl -fsS https://$MEDHA_DOMAIN/healthz
   curl -fsS https://$MEDHA_DOMAIN/readyz
   ```

   `/healthz` checks the web process; `/readyz` also checks the configured brain. Do not open signup until readiness succeeds and the owner has verified the selected model.

## Staging first

Use a separate staging hostname, `.env.staging`, and Compose project so volumes and invite codes stay separate:

```sh
cp .env.example .env.staging
docker compose -p medha-staging --env-file .env.staging up -d --build
docker compose -p medha-staging --env-file .env.staging ps
```

Set `MEDHA_DOMAIN` in `.env.staging` to the staging host. Exercise signup/consent, search, export, deletion, maintenance mode, a backup, and a restore before changing production DNS.

## Backup and restore

The Compose `backup` service makes a verified SQLite snapshot on startup and every 24 hours, retaining the latest seven snapshots under `/data/backups` on the persistent volume. Check its logs and list the files before relying on a backup. This is local rotation, not offsite protection; the owner must configure encrypted offsite copies and retention.

To validate and restore a snapshot without overwriting the live database, stop the app and restore to a new path:

```sh
docker compose stop app
docker compose run --rm --no-deps app python scripts/backup.py restore \
  --backup /data/backups/<backup-file> --target /data/medha-restored.sqlite3
```

Check the restore output, take a separate copy of the existing database, then follow the owner-reviewed file swap procedure. The restore command refuses to replace any existing target. Start the app and verify `/healthz`, `/readyz`, sign-in, and sample conversation access before reopening traffic.

## Rollback

1. Close signup and writes with the admin flags or set `MEDHA_MAINTENANCE=true` in `.env` and recreate the app container.
2. Check out the last known-good code revision, rebuild the image, and recreate the app: `docker compose up -d --build app`.
3. If the release changed user data and rollback is necessary, restore a verified snapshot to a new path first. Do not overwrite the only current database or backup.
4. Verify health, readiness, consent/signup, search citations, and export on staging before reopening production.
5. Record the revision, backup selected, recovery time, and any follow-up incident actions without copying user content into the record.

## Host limitations

The Compose stack is a single worker and a single SQLite host. The in-memory limiter is not shared across processes. Choose the user cap and model only after testing load and capability on the target hardware. Keep the model endpoint private; do not publish Ollama's port. Coordinate dependency and base-image updates through CI and the launch checklist.
