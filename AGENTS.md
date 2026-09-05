# Engineering contract

- Support Python 3.11 and 3.12 and keep application code under `src/gpt2lab`.
- Preserve deterministic, offline unit tests. Never make tests download data or
  models, require an accelerator, or write outside pytest temporary directories.
- Keep configuration immutable and validate invalid states at construction.
- Keep external boundaries injectable: corpus download, tokenization, device
  selection, clocks, and checkpoint persistence.
- Keep CLI functions thin; business logic belongs in importable modules.
- Add type hints to production code and keep strict mypy clean.
- Run `make quality` before publishing. Do not claim a check passed unless its
  command completed successfully.
- Never commit corpora, model weights, checkpoints, credentials, caches, logs,
  local environments, or generated plots.
- Use domain-specific exceptions with actionable messages at package boundaries.
- Prefer small focused changes and update tests and documentation with behavior.
