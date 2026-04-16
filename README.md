# SciArx

SciArx is a scientific computing library for ArxLang.

This repository currently contains the initial project skeleton only.
The implementation will follow the roadmap in phased steps, beginning
with `sciarx.core`.

## Status

Early bootstrap stage. No stable numerical APIs yet.

## Planned modules

- `sciarx.core`
- `sciarx.linalg`
- `sciarx.signal`
- `sciarx.stats`

See `ROADMAP.md` for the full plan.

## Tests

This directory contains the initial placeholder test files for SciArx.

At this stage, SciArx does not yet define a full test stack or assertion API.
These files exist to establish an early test layout and to exercise simple
module imports:

- `test_lib.arx`
- `test_version.arx`
- `core/test_core_import.arx`

Later, these files can be migrated to the real test harness once the ArxLang
testing workflow is defined.

## ArxPM

This repository now includes a minimal `arxproj.toml` manifest for `arxpm`.

Current intent:
- package/library-style repository
- manifest entry points to `src/sciarx/lib.arx`
- build artifacts go to `build/`

Useful future commands may include:
- `arxpm doctor`
- `arxpm install`
- `arxpm build`
- `arxpm pack`

If later SciArx needs an executable smoke target, add a dedicated
`src/main.x` and update `[build].entry` accordingly.
