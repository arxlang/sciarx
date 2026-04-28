```
title: SciArx math helpers
summary: Numeric scalar helpers that complement built-in arithmetic operators.
```

type Numeric = i8 | i16 | i32 | i64 | f32 | f64

@<T: Numeric>
fn min(lhs: T, rhs: T) -> T:
  ```
  title: min
  summary: Returns the smaller of two scalar values.
  ```
  if lhs < rhs:
    return lhs
  else:
    return rhs

@<T: Numeric>
fn max(lhs: T, rhs: T) -> T:
  ```
  title: max
  summary: Returns the larger of two scalar values.
  ```
  if lhs > rhs:
    return lhs
  else:
    return rhs

@<T: Numeric>
fn clip(value: T, lower: T, upper: T) -> T:
  ```
  title: clip
  summary: Clamps one scalar value to an inclusive interval.
  ```
  if value < lower:
    return lower
  else:
    if value > upper:
      return upper
    else:
      return value

@<T: Numeric>
fn sign(value: T) -> i32:
  ```
  title: sign
  summary: Returns -1, 0, or 1 for one scalar value.
  ```
  var zero: T = cast(0, T)
  if value < zero:
    return 0 - 1
  else:
    if value > zero:
      return 1
    else:
      return 0

@<T: Numeric>
fn abs(value: T) -> T:
  ```
  title: abs
  summary: Returns the absolute value of one numeric scalar.
  ```
  var zero: T = cast(0, T)
  if value < zero:
    return zero - value
  else:
    return value
