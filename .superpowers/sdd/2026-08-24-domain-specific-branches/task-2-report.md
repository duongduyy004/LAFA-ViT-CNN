# Task 2 report: emit branch-input mappings

## Status

Implemented the Task 2 data-pipeline contract on master in the shared workspace. RGB is always emitted first, followed by optional SRM and FFT mappings in canonical order. Legacy wavelet and concatenated-artifact helpers were removed from the scoped data module.

## TDD evidence

### RED

Command:

~~~text
rtk .venv/bin/python -m pytest tests/test_data.py -q
~~~

Result: collection failed as expected because build_branch_inputs did not yet exist. The failure was:

~~~text
ImportError: cannot import name 'build_branch_inputs' from 'favit_lsda.data'
~~~

### GREEN

The focused behavior suite passed after implementation:

~~~text
rtk .venv/bin/python -m pytest tests/test_data.py -q
......................                                                   [100%]
~~~

Because the unchanged, out-of-scope favit_lsda/model.py still imports the intentionally removed artifact_channels, this run used a temporary, non-committed compatibility shim solely to isolate Task 2 behavior. The shim was removed immediately after the run.

Additional checks:

- rtk .venv/bin/python -m py_compile favit_lsda/data.py tests/test_data.py: passed.
- rtk git diff --check: passed.
- Legacy helper search in the two scoped files found no artifact_channels, build_cnn_input, artifact_mode, wavelet, layout, or resolver references.

## Files changed

- favit_lsda/data.py
  - Added build_branch_inputs(rgb, enable_srm, enable_fft, sample_path).
  - Kept finite RGB validation, constant-frame detection, SRM/FFT builders, and artifact normalization.
  - Updated FaceTransform to store enable flags and return an ordered mapping.
  - Updated grouped LSDA samples to validate common branch keys and stack each mapping independently.
  - Updated frame samples to return (branch_inputs, label, video_id).
  - Removed _ARTIFACT_LAYOUTS, artifact_channels, resolve_artifact_config, _wavelet_artifact, and build_cnn_input.

- tests/test_data.py
  - Replaced mode/width and concatenated-tensor assertions with exact mapping-key, shape, finite-value, ordering, constant-artifact, transform, grouped-dataset, and frame-dataset tests.
  - Removed wavelet and legacy helper coverage.

## Self-review

- Mapping order is deterministic: rgb, then srm when enabled, then fft when enabled.
- Disabled builders are not invoked.
- Every emitted branch retains shape [3, H, W]; grouped stacking produces [domains, 3, H, W].
- Group key inconsistency raises the required explicit error.
- No model, engine, or config files were modified.

## Concerns

The required unshimmed full-suite command:

~~~text
rtk .venv/bin/python -m pytest -q
~~~

cannot collect six modules because unchanged downstream favit_lsda/model.py imports deleted artifact_channels. This is the expected Task 2 migration boundary and is intentionally out of scope per the brief; Task 4/model migration must update those consumers. The full-suite run therefore did not reach test execution.
