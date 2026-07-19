# Water Potability — EDA Summary
- Rows: **3276**, Columns: **10**

## Missing values
| Column | Missing | % |
|---|---|---|
| ph | 491 | 14.99% |
| Sulfate | 781 | 23.84% |
| Trihalomethanes | 162 | 4.95% |

## Target balance (Potability)
- Not potable (0): 1998 (61.0%)
- Potable (1): 1278 (39.0%)

**Note:** the dataset is moderately imbalanced (~61/39), so we use `class_weight="balanced"` and report precision/recall/F1 per class, not just accuracy.

## Summary statistics
|                 |   count |     mean |     std |    min |      25% |      50% |      75% |      max |
|:----------------|--------:|---------:|--------:|-------:|---------:|---------:|---------:|---------:|
| ph              |    2785 |     7.08 |    1.59 |   0    |     6.09 |     7.04 |     8.06 |    14    |
| Hardness        |    3276 |   196.37 |   32.88 |  47.43 |   176.85 |   196.97 |   216.67 |   323.12 |
| Solids          |    3276 | 22014.1  | 8768.57 | 320.94 | 15666.7  | 20927.8  | 27332.8  | 61227.2  |
| Chloramines     |    3276 |     7.12 |    1.58 |   0.35 |     6.13 |     7.13 |     8.11 |    13.13 |
| Sulfate         |    2495 |   333.78 |   41.42 | 129    |   307.7  |   333.07 |   359.95 |   481.03 |
| Conductivity    |    3276 |   426.21 |   80.82 | 181.48 |   365.73 |   421.88 |   481.79 |   753.34 |
| Organic_carbon  |    3276 |    14.28 |    3.31 |   2.2  |    12.07 |    14.22 |    16.56 |    28.3  |
| Trihalomethanes |    3114 |    66.4  |   16.18 |   0.74 |    55.84 |    66.62 |    77.34 |   124    |
| Turbidity       |    3276 |     3.97 |    0.78 |   1.45 |     3.44 |     3.96 |     4.5  |     6.74 |
