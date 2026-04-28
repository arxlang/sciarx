```
title: SciArx math tests
summary: Covers scalar numeric helpers and Arx type alias smoke paths.
```

import sciarx.math as math

type TestNumber = i32 | i64

fn test_math_scalar_helpers() -> none:
  ```
  title: test_math_scalar_helpers
  summary: Verifies numeric scalar helpers.
  ```
  assert math.min(3, 9) == 3
  assert math.max(3, 9) == 9
  assert math.clip(12, 0, 10) == 10
  assert math.sign(0 - 4) == (0 - 1)
  assert math.abs(0 - 5) == 5
  assert math.abs(1.0 - 3.5) == 2.5

fn test_math_type_alias_builtins() -> none:
  ```
  title: test_math_type_alias_builtins
  summary: Verifies type aliases, isinstance, and cast support.
  ```
  var value: i64 = cast(5, i64)
  var ok: bool = isinstance(value, TestNumber)
  var narrowed: i32 = cast(value, i32)
  assert ok == true
  assert narrowed == 5
