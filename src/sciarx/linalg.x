```
title: SciArx linear algebra
summary: Linear algebra scalar kernels and tensor entry-point placeholders.
```

type Numeric = i8 | i16 | i32 | i64 | f32 | f64
type Float = f32 | f64

@<T: Numeric>
fn dot_pair(lhs_first: T, lhs_second: T, rhs_first: T, rhs_second: T) -> T:
  ```
  title: dot_pair
  summary: Computes a two-term dot-product kernel without exposing tensor size in the API name.
  ```
  return (lhs_first * rhs_first) + (lhs_second * rhs_second)

@<T: Numeric>
fn trace_pair(first_diagonal: T, second_diagonal: T) -> T:
  ```
  title: trace_pair
  summary: Computes a two-term trace kernel without exposing tensor size in the API name.
  ```
  return first_diagonal + second_diagonal

@<T: Numeric>
fn determinant(top_left: T, top_right: T, bottom_left: T, bottom_right: T) -> T:
  ```
  title: determinant
  summary: Computes a small dense determinant kernel.
  ```
  return (top_left * bottom_right) - (top_right * bottom_left)

@<F: Float>
fn solve_first(top_left: F, top_right: F, bottom_left: F, bottom_right: F, rhs_top: F, rhs_bottom: F) -> F:
  ```
  title: solve_first
  summary: Returns the first component of a small dense linear solve kernel.
  ```
  var det: F = determinant(top_left, top_right, bottom_left, bottom_right)
  return ((rhs_top * bottom_right) - (top_right * rhs_bottom)) / det

@<F: Float>
fn solve_second(top_left: F, top_right: F, bottom_left: F, bottom_right: F, rhs_top: F, rhs_bottom: F) -> F:
  ```
  title: solve_second
  summary: Returns the second component of a small dense linear solve kernel.
  ```
  var det: F = determinant(top_left, top_right, bottom_left, bottom_right)
  return ((top_left * rhs_bottom) - (rhs_top * bottom_left)) / det

@<F: Float>
fn norm_pair(first: F, second: F) -> F:
  ```
  title: norm_pair
  summary: Computes a two-term absolute-value norm kernel without exposing tensor size in the API name.
  ```
  if first < 0.0:
    if second < 0.0:
      return (0.0 - first) + (0.0 - second)
    else:
      return (0.0 - first) + second
  else:
    if second < 0.0:
      return first + (0.0 - second)
    else:
      return first + second

fn accept_vector(values: tensor[f64, ...]) -> none:
  ```
  title: accept_vector
  summary: Accepts a runtime-shaped f64 tensor for future vector routines.
  ```
  return none

fn accept_matrix(values: tensor[f64, ...]) -> none:
  ```
  title: accept_matrix
  summary: Accepts a runtime-shaped f64 tensor for future matrix routines.
  ```
  return none
