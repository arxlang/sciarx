# SciArx

SciArx is a scientific computing library for ArxLang.

This repository contains the initial SciArx implementation for the current Arx
toolchain. The implementation follows the roadmap in phased steps, beginning
with focused math, stats, and linalg modules.

## Status

Early bootstrap stage. `sciarx.math` includes numeric scalar helpers,
`sciarx.stats` includes scalar statistical kernels, and `sciarx.linalg` includes
initial dense scalar kernels plus runtime-shaped tensor entry points. Arx
collection types such as `tensor`, `dataframe`, and `series` are covered by
compatibility tests, but SciArx does not expose placeholder wrapper modules for
them yet.

Full tensor-returning APIs such as `zeros`, `sum`, `matmul`, and `solve` are
specified in `docs/phase-0-2-proposal.md`, but remain gated on Arx source
support for unsized tensor values outside parameter positions.

## Planned modules

- `sciarx.math`
- `sciarx.stats`
- `sciarx.linalg`
- `sciarx.signal`

See `ROADMAP.md` for the full plan.

## Tests

The test suite uses the Arx test runner:

```bash
arx test tests
```

Current tests cover Phase 0 collection smoke paths, Phase 1 math/stats helpers,
Phase 2 linalg kernels, and the `sciarx.stats` smoke helpers.

## ArxPM

This repository includes a minimal `.arxproject.toml` manifest for `arxpm`.

Current intent:

- package/library-style repository
- sources live under `src/`
- build artifacts go to `build/`

Useful future commands may include:

- `arxpm doctor`
- `arxpm install`
- `arxpm build`
- `arxpm pack`

If later SciArx needs an executable smoke target, add a dedicated `src/main.x`
and update `[build].entry` accordingly.
