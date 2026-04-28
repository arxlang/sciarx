```
title: SciArx collection compatibility tests
summary: Covers Arx tensor, DataFrame, and Series types used by SciArx modules.
```

fn accept_i32_tensor(values: tensor[i32, ...]) -> none:
  ```
  title: accept_i32_tensor
  summary: Accepts a runtime-shaped i32 tensor.
  ```
  return none

fn dataframe_rows(rows: dataframe[...]) -> i64:
  ```
  title: dataframe_rows
  summary: Returns row count for a runtime-schema DataFrame.
  ```
  return rows.nrows()

fn dataframe_cols(rows: dataframe[...]) -> i64:
  ```
  title: dataframe_cols
  summary: Returns column count for a runtime-schema DataFrame.
  ```
  return rows.ncols()

fn accept_f64_series(values: series[f64]) -> none:
  ```
  title: accept_f64_series
  summary: Accepts a typed f64 Series.
  ```
  return none

fn test_tensor_type_acceptance() -> none:
  ```
  title: test_tensor_type_acceptance
  summary: Verifies shaped tensors can flow into runtime-shaped parameters.
  ```
  var ids: tensor[i32, 3] = [1, 2, 3]
  accept_i32_tensor(ids)
  assert ids[2] == 3

fn test_dataframe_and_series_type_acceptance() -> none:
  ```
  title: test_dataframe_and_series_type_acceptance
  summary: Verifies runtime-schema DataFrame and typed Series support.
  ```
  var rows: dataframe[id: i32, score: f64] = dataframe({
    id: [1, 2, 3],
    score: [0.5, 0.8, 1.0],
  })
  var scores: series[f64] = rows.score
  accept_f64_series(scores)
  assert dataframe_rows(rows) == 3
  assert dataframe_cols(rows) == 2
