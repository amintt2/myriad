# Contributing to Myriad

Thanks for your interest! Myriad is young: bug reports, measurements on your hardware, translations
and code are all welcome. *Les contributions en français sont les bienvenues.*

## Ground rules

- **Code and code comments in English; user-facing text and documentation in French first.** Every
  string shown in the interface goes through the FR/EN dictionary in `app/myriad/web/i18n.js` (add both
  languages). Commit messages are written in French in this repository; English is accepted.
- **Only open-weight models under a permissive licence (Apache-2.0 or MIT), in GGUF/safetensors.** No
  `trust_remote_code`. A new catalogue entry (`app/myriad/catalog.py`) must pin the repository
  revision, the file size and its SHA-256 (from `https://huggingface.co/api/models/<repo>?blobs=true`).
- **Never commit a secret** (keys, tokens, passwords, `node_key.pem`, `config.json`).
- The interface loads nothing from the internet (no CDN): vendor small libraries with a permissive
  licence, or write vanilla JS.
- Keep the protocol backward compatible, or bump its version (`app/myriad/__init__.py`) and explain why.

## Development setup

```bash
cd app
uv sync --extra desktop --group build
uv run pytest -q                         # ~40 s, no GPU and no network needed
uv run python scripts/demo_swarm.py      # a local demo network for the UI
uv run myriad-desktop --home /tmp/myriad-dev
```

Tests with a real model: `MYRIAD_TEST_GGUF=/path/small.gguf LLAMA_SERVER=/path/llama-server uv run
pytest -q -m real`.

## Pull requests

1. One topic per pull request, with tests for new behaviour (`app/tests/`).
2. `uv run pytest -q` passes.
3. For UI changes, attach a screenshot (`uv run python scripts/screenshots.py` regenerates
   `app/docs/`), check both themes, both languages and keyboard navigation.
4. Code review: the maintainers also run an automated review (Codex) on each change; each remark is
   checked by hand before being applied or rejected.

## Releases

Tag `vX.Y.Z` (matching `__version__` in `app/myriad/__init__.py`): `.github/workflows/release.yml`
tests, builds Windows / macOS / Linux packages, smoke-tests each one and opens a **draft** release.
Signing secrets (Authenticode, Apple Developer ID and notarization) are optional and documented at the
top of the workflow.

## Licence

By contributing, you agree that your contributions are licensed under the Apache License 2.0.
