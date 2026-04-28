```
title: SciArx linear algebra tests
summary: Covers Phase 2 scalar kernels and runtime-shaped tensor entry points.
```

import sciarx.linalg as linalg

fn test_linalg_scalar_kernels() -> none:
  ```
  title: test_linalg_scalar_kernels
  summary: Verifies dense linear algebra scalar kernels.
  ```
  assert linalg.dot_pair(1.0, 2.0, 3.0, 4.0) == 11.0
  assert linalg.trace_pair(5, 7) == 12
  assert linalg.determinant(1.0, 2.0, 3.0, 4.0) == (1.0 - 3.0)
  assert linalg.solve_first(1.0, 0.0, 0.0, 1.0, 5.0, 7.0) == 5.0
  assert linalg.solve_second(1.0, 0.0, 0.0, 1.0, 5.0, 7.0) == 7.0
  assert linalg.norm_pair(1.0 - 4.0, 4.0) == 7.0

fn test_linalg_runtime_tensor_acceptance() -> none:
  ```
  title: test_linalg_runtime_tensor_acceptance
  summary: Verifies runtime-shaped tensor entry points for future vector and matrix APIs.
  ```
  var values: tensor[f64, 4] = [1.0, 2.0, 3.0, 4.0]
  linalg.accept_vector(values)
  linalg.accept_matrix(values)
  assert values[0] == 1.0
