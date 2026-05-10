# Agent Context

## Project

Dog Breed Image Classification System.

The app accepts uploaded dog images, predicts one of 9 breeds, stores the uploaded
image, records prediction metadata, and loads prediction history into PostgreSQL.

## Engineering Style

Write code for a small production system, not a demo notebook.

- Keep code concise and minimalist.
- Prefer plain functions and simple data flow.
- Avoid abstractions until there is repeated complexity to remove.
- Do not add frameworks, services, or dependencies unless they are clearly needed.
- Document all code with short docstrings or comments where intent is not obvious.
- Comments should explain why code exists, not restate what each line does.
- Use explicit names over clever code.
- Keep functions small and focused.
- Preserve traceability for data and model outputs.
- Add or update tests for behavior changes.

# Workflow Preferences

When the user asks to push changes, treat it as a PR publishing workflow unless they explicitly say otherwise:

1. Create or use a feature branch. Do not work directly on `main` for publishable changes.
2. Keep edits scoped to the requested change and avoid unrelated cleanup.
3. Run relevant local tests/checks. Prefer the local virtualenv:

   ```powershell
   myvenv\Scripts\python.exe -m pytest
   ```

4. Commit the current changes with a clear message.
5. Push the current branch.
6. If there is no open PR for the branch, open a draft PR against `main`.
7. If a PR already exists, update it and report the PR URL.
8. Add this PR comment to request an independent Codex review:

   ```text
   @codex please review this PR for correctness, regressions, maintainability, and missing tests.
   ```

9. Fetch PR comments and review threads after the review has had time to run.
10. Address actionable review feedback, rerun tests, commit fixes, and push follow-up commits.
11. Leave the PR as draft unless the user asks to mark it ready for review.

Treat review comments as actionable only when they identify a concrete correctness, maintainability, security, performance, or test-coverage issue. Do not make speculative changes just to satisfy vague feedback; ask for clarification or explain why no code change is needed.

## Current Architecture

### API

Framework: FastAPI.

Responsibilities:

- Accept image uploads at `/predict`.
- Validate uploaded files.
- Run model inference.
- Return prediction, confidence, model version, timestamp, and prediction id.
- Save uploaded images under `storage/raw_uploads/`.
- Append prediction metadata to `storage/predictions/predictions.jsonl`.

### Prediction Log

Format: JSON Lines.

Each prediction is one JSON object per line. New predictions append to the file.
The log is an audit source and should not be rewritten casually.

Required fields:

- `id`
- `uploaded_filename`
- `predicted_breed`
- `confidence`
- `model_version`
- `timestamp`
- `image_path`

### Metadata Ingestion

The ingestion job reads `storage/predictions/predictions.jsonl` and writes rows to
PostgreSQL using SQLAlchemy.

Each database row must be backtraceable to the source log:

- `source_log_path`
- `source_line_number`
- `source_raw_json`
- `ingested_at`

The ingestion job should be idempotent. Running it twice must not duplicate rows.

## Core Rules

- Never train on unreviewed uploads.
- Store image files separately from metadata.
- Keep prediction history auditable.
- Always preserve model version and timestamp.
- Prefer PostgreSQL for durable metadata storage.
- Keep storage paths relative when recorded by the API.

## Future Scope

Manual annotation, dataset versioning, retraining, evaluation, and deployment are
future steps. Build them incrementally and keep each step testable.
