---
name: Bug report
about: Report a problem with YAPA
title: "[bug] "
labels: bug
assignees: ""
---

## Description

A clear and concise description of the bug.

## Steps to reproduce

1. Configure provider: `yapa config set provider_configs.openai.api_key sk-...`
2. Start the server: `yapa server`
3. Send request: ...
4. See error: ...

## Expected behavior

What you expected to happen.

## Actual behavior

What actually happened. Include the full error message or traceback.

## Environment

- YAPA version (commit hash or release tag):
- Python version:
- OS / platform:
- Provider(s) used:
- Invocation method: CLI / WebSocket / REST / installed `yapa` / `uv run python -m yapa`

## Logs

Relevant lines from the log file at `~/.yapa/logs/{YYYY-MM-DD}/{name}.log`. Redact any API keys.

## Additional context

Anything else that might help, e.g. config (without secrets), repro scripts, or screenshots.