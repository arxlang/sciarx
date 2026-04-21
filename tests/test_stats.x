```
title: Main test file
```

import sciarx.stats as stats

fn test_sum2() -> none:
    assert stats.sum2(1, 2) == 3, "sum2 is incorrect"

fn test_mean2() -> none:
    assert stats.mean2(1, 2) == 1.5, "mean2 is incorrect"
