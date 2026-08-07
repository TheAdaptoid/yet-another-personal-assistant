# Plan: Fix release blockers #23, #24, #26, #27

Date: 2026-08-07
Branch: `bug-bounty`
Baseline: HEAD `cb812a0` (after #19/#20/#21/#22/#25 fixes, all closed)

## Issues

| Issue | Title | Root cause (verified in code) |
|---|---|---|
| #23 | `ModelService.get_model` split(':', 1) truncates provider model IDs that contain colons | TBD — must be verified first. `split(":", 1)` on the first colon preserves everything after it, so the truncation claim is suspicious. Possible real repro: a full id whose *model* portion contains `:` (e.g. Ollama `llama3.2:8b`) fed through `GET /models/{full_id:path}` vs. service-level parse. Strict TDD: write failing test first; only implement if a test genuinely fails. |
| #24 | `OpenAIIP._format_model` `or` chain overwrites valid falsy values (0) | `src/yapa/providers/openai/provider.py` `model_copy(update={...})` uses `name or meta.get(...) or base.name`, `context_length or ...`, `max_output or ...` — a native `0` for `max_output`/`context_length` (or empty `name`) falls through to metadata/base. Replace `or` chains with `is not None` checks for the four scalar fields (name, description, context_length, max_output). pricing is an object (truthy) — leave as-is but confirm. |
| #26 | Approval callback exceptions bubble to `AgentErrorEvent` | `src/yapa/services/chat.py` `_execute_tool_calls`: `await get_approval(...)` is unguarded around line 326. Per spec (`docs/specs/2026-07-27-agentic-loops-design.md:106`) callback failures (timeout/disconnect) must be caught → denial `ToolMessage` → loop continues. Wrap in try/except (TimeoutError → "Error: approval timeout"), continue loop. |
| #27 | `MAX_ITERATIONS` reached discards the in-flight turn | `src/yapa/services/chat.py` line 159: `yield AgentErrorEvent(message="Max iterations reached")` returns without persisting. Reuse `_finalize_turn`-style persistence (add_messages with user + accumulated messages) before yielding the terminal error. |

## Process (strict TDD, same as previous batch)

For each issue, one sub-agent:

1. RED — write the failing test(s) in the matching tests dir; run the file; confirm the test fails for the right reason.
2. GREEN — minimal implementation change to make it pass.
3. Run the touched test file + adjacent suite, then a full `uv run pytest tests/ -q --no-cov -p no:cacheprovider` gate with `uv run ruff check src/ tests/` and `uv run ty check src/`.
4. Do NOT commit; report diff summary.

## Notes

- `#26` interacts with existing #19 fix (`get_approval is None` → deny ToolMessage). The new branch is for *raising* callbacks.
- `#27`: persist at minimum `user_msg` + accumulated assistant/tool messages of this turn before emitting the error, consistent with `add_messages(msg, model=model)`.
- `#23`: if the strict failing test cannot be reproduced (parse already preserves model colons), report the finding and the evidence; do not invent a change. If a real repro exists (e.g. route-level), fix it minimally (e.g. `split(":", 1)` is right-side; or document provider-less parse; or rsplit) and add regression tests.

## Gate

`uv run ruff check src/ tests/ && uv run ty check src/ && uv run pytest tests/ -v`

Commit per issue with `fix:` prefix; close issue with completion comment referencing commit.