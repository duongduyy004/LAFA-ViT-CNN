# Task 6 report: version multibranch checkpoints

## Commit

- `68f5a00 feat: version multibranch checkpoint metadata`

## Changes

- Added v4 `favit_lsda_multibranch` metadata and model-authoritative
  `model_branch_metadata`.
- Added strict branch/backbone/fusion compatibility validation with checkpoint,
  config, and path diagnostics.
- Added explicit format-v3/legacy migration guidance via `--init-favit`.
- Updated resume and evaluation validation; evaluation now loads state strictly.
- Excluded `head.`, `srm_encoder.`, `fft_encoder.`, and `late_fusion.` from
  FA-ViT initialization.

## Evidence

- RED: `tests/test_checkpoint.py` failed against the pre-v4 implementation due
  to the missing validator and stale v3 metadata expectations.
- GREEN: `tests/test_checkpoint.py tests/test_engine.py -q` passed.
- Regression: `.venv/bin/python -m pytest -q -rA` passed at 100%.
- `git diff --check` passed.

## Issues

- The sandboxed rtk path emitted the known loopback warning; verification was
  run through the approved escalated read-only/test path. No test failures or
  unresolved Task 6 issues remain.
