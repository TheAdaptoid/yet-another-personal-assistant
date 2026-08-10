# Session PATCH Clear Semantics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the session system-prompt and inference-params PATCH endpoints require explicit fields and use JSON `null` as the documented clear operation.

**Architecture:** Add small Pydantic request models for the two PATCH payloads, with required nullable fields so Pydantic/FastAPI can distinguish an omitted field from an explicit `null`. Keep `SessionService` unchanged: the routes pass either the supplied value or `None`, while `InferenceParams` continues validating the nested parameter object.

**Tech Stack:** Python 3.13, FastAPI, Pydantic v2, pytest.

## Global Constraints

- Do not add SQLite/sqlmodel dependencies.
- Preserve the existing `SessionService.update_system_prompt()` and `update_inference_params()` interfaces.
- Follow the repository gate: `uv run ruff check src/ tests/ && uv run ty check src/ && uv run pytest tests/ -v`.

---

### Task 1: Add typed PATCH request models and route wiring

**Files:**
- Modify: `src/yapa/api/routes/sessions.py`

**Interfaces:**
- Produce `SystemPromptPatch` with required field `system_prompt: str | None`.
- Produce `InferenceParamsPatch` with required field `inference_params: InferenceParams | None`.
- The routes continue calling `update_system_prompt(session_id, body.system_prompt)` and `update_inference_params(session_id, body.inference_params)`.

- [ ] **Step 1: Define the request models**

  Import `BaseModel` and add module-level models with no default for the wrapped fields. In Pydantic, `str | None` and `InferenceParams | None` without a default are nullable but required, so `{}` produces FastAPI's normal 422 response while `null` remains valid.

  ```python
  class SystemPromptPatch(BaseModel):
      """Request body for setting or clearing a session system prompt."""

      system_prompt: str | None


  class InferenceParamsPatch(BaseModel):
      """Request body for setting or clearing session inference parameters."""

      inference_params: InferenceParams | None
  ```

- [ ] **Step 2: Replace raw dictionary bodies**

  Change each endpoint annotation from `body: dict` to the corresponding request model and remove `.get()`/manual construction. This makes the accepted JSON explicit:

  ```python
  prompt = body.system_prompt
  return session_service.update_system_prompt(str(session_id), prompt)
  ```

  ```python
  return session_service.update_inference_params(
      str(session_id), body.inference_params
  )
  ```

- [ ] **Step 3: Run focused route tests**

  Run: `uv run pytest tests/api/test_sessions.py -v`

  Expected: existing set/clear tests pass except the inference clear test, which should be updated in Task 2 because its request shape changes.

### Task 2: Lock down omission, null, and value behavior with API tests

**Files:**
- Modify: `tests/api/test_sessions.py`

**Interfaces:**
- Tests exercise the public HTTP contract and verify the mocked service receives the intended `None` or typed value.

- [ ] **Step 1: Update explicit clear payloads**

  Change system-prompt clear to `json={"system_prompt": None}` if needed (it already matches the new contract). Change inference clear from `{}` to `{"inference_params": None}` and assert the service receives `None`.

- [ ] **Step 2: Add omission rejection tests**

  Add tests for `{}` against both PATCH endpoints. Assert status `422` and assert the corresponding service mock was not called. These reproduce the bug's missing-field cases.

- [ ] **Step 3: Add nested inference payload coverage**

  Keep the existing temperature test but change its body to `{"inference_params": {"temperature": 0.7}}`; assert the service receives an `InferenceParams` with `temperature == 0.7`. Add a malformed nested value test if not already covered elsewhere, asserting `422`, to prove nested Pydantic validation remains active.

- [ ] **Step 4: Run focused tests**

  Run: `uv run pytest tests/api/test_sessions.py -v`

  Expected: all session API tests pass, including set, explicit null clear, and omitted-field 422 cases.

### Task 3: Document the request contract

**Files:**
- Modify: `README.md:127-130`

**Interfaces:**
- Document the JSON shapes clients must send; no runtime interface changes.

- [ ] **Step 1: Replace endpoint-only descriptions with payload examples**

  Document:

  ```json
  {"system_prompt": "Be concise."}
  {"system_prompt": null}
  {"inference_params": {"temperature": 0.7}}
  {"inference_params": null}
  ```

  State that omitting the required wrapper field (including `{}`) returns `422`, and that `null` is the explicit clear operation.

- [ ] **Step 2: Run the complete verification gate**

  Run: `uv run ruff check src/ tests/ && uv run ty check src/ && uv run pytest tests/ -v`

  Expected: lint, type checking, and the full suite pass.

- [ ] **Step 3: Review the diff for scope**

  Run: `git diff --check; git diff -- src/yapa/api/routes/sessions.py tests/api/test_sessions.py README.md`

  Confirm the change is limited to request validation, regression tests, and API documentation; do not modify service/storage behavior.

## Self-Review Checklist

- The issue's omission cases are covered by explicit 422 tests.
- Both clear operations use JSON `null` and are passed to the existing service methods as `None`.
- Non-null inference parameters remain validated by `InferenceParams`.
- No endpoint relies on `dict.get()` for required PATCH fields.
- README examples match the exact request models.
