# Contributor rules

- Prefer the Python standard library. Any new dependency must be pinned and justified in `docs/DEPENDENCIES.md`. Allowed if needed: `httpx`, `python-multipart`, and `pypdf`. Ask before adding anything else.
- Write readable, typed code and explain why in comments where the reason is not obvious.
- Make one commit per work package, with a message that explains why. Run all test suites before every commit.
- Do not use live network in tests. Use fixtures and fakes. Do not commit secrets, model weights, or databases; provide `.env.example` only.
- Add a test for every security rule so removing the rule makes a test fail.
- Never log full message text, passwords, tokens, or uploaded file contents.
- If a decision is not covered by the task, choose the simplest safe option, record it in `docs/DECISIONS.md`, and continue.
- Stop and ask the owner only before adding paid services, irreversible production data deletion, or final legal wording.
