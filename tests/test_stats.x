```
title: Main test file
```

import sciarx.stats as stats

fn test_sum() -> none:
    assert stats.sum(1, 2) == 3, "sum is incorrect"

fn test_mean() -> none:
    assert stats.mean(1.0, 2.0) == 1.5, "mean is incorrect"

fn test_variance() -> none:
    assert stats.variance(2.0, 4.0) == 1.0, "variance is incorrect"
