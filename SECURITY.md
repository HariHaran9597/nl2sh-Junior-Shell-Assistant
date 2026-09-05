# Security policy

## Scope

`nl2sh+` generates shell command suggestions. It must not be treated as a
security boundary, command approval system, or autonomous shell agent.

The public Gradio Space is generation-only and does not execute generated
commands. Review every command before running it. The local CLI requires the
explicit `--execute` flag and refuses commands matched as DANGER, but its
classifier is heuristic and cannot guarantee safety.

## Reporting

Please report security issues privately through the repository owner's GitHub
security contact rather than opening a public issue with an exploitable
payload. Include the affected file, reproduction steps, and impact.

## Deployment requirements

- Never expose a local llama.cpp server without authentication and restricted
  CORS.
- Do not put API tokens in source, notebooks, model cards, or Git history.
- Run generated commands only in an isolated, non-root environment with a
  restricted working directory and no network access.
