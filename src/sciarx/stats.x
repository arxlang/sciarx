```
title: SciArx statistics helpers
summary: Statistical scalar kernels that will back tensor and Series reductions.
```

type Numeric = i8 | i16 | i32 | i64 | f32 | f64
type Float = f32 | f64

@<T: Numeric>
fn sum(a: T, b: T) -> T:
    ```
    title: sum
    summary: Returns the sum of two numeric values.
    ```
    return a + b

@<F: Float>
fn mean(a: F, b: F) -> F:
    ```
    title: mean
    summary: Returns the arithmetic mean of two floating-point values.
    ```
    return (a + b) / 2.0

@<F: Float>
fn variance(a: F, b: F) -> F:
    ```
    title: variance
    summary: Returns the population variance of two floating-point values.
    ```
    var center: F = mean(a, b)
    var left: F = a - center
    var right: F = b - center
    return ((left * left) + (right * right)) / 2.0
