# Security Policy

## Supported versions

Only the latest released minor version receives security fixes while the project is pre-1.0.

## Reporting a vulnerability

Please **do not open a public issue**. Use GitHub's private vulnerability reporting ("Security" tab → "Report a vulnerability") on this repository. Include a description, affected versions, reproduction steps and impact. We aim to acknowledge reports within 7 days and to publish a fix or mitigation within 90 days, coordinating disclosure with you.

## Scope notes

tsuzuri stores conversation content on local disk (SQLite and JSON under the data directory). Deployments are responsible for:

- protecting the data directory (it contains personal conversation data);
- keeping API keys out of the repository (use environment variables or a `tsuzuri.toml` in the data directory, never committed);
- treating recalled memory as untrusted input to the model — memory rows were written from user conversations and may contain prompt-injection attempts.
