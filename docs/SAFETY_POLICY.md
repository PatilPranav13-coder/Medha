# Medha safety policy

Medha is a research and coding assistant. It should decline instructions that enable severe harm, including weapon construction, poisoning, malware theft, violent attack planning and sabotage. It may redirect to safety, prevention, defensive security or benign educational context.

When a message indicates self-harm or suicide risk, Medha returns a supportive response without calling the configured brain. The response points to Tele-MANAS in India: **14416** or **1-800-891-4416**, free, 24x7. The phone numbers and availability live in `safety/resources.json` so the owner can update them.

Web results and future uploaded-file text are untrusted data. The prompt builder encodes them inside bounded delimiters and explicitly tells the brain not to follow instructions inside that data. Citation references from synthesis are checked against the available source list; invalid citations fall back to a verbatim extractive response.

Request bodies, chat rates, conversation length, active generation and queued work have configurable limits. These controls protect availability; they do not guarantee abuse prevention. Review red-team results and incident reports before increasing the beta user cap.
