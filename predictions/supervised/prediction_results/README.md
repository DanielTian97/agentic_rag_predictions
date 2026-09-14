# Supervised predictor results

Put generated supervised-regressor prediction CSV files here.

By default, `predict.py` writes one file per relation and target, for example:

```text
adjacent_think_performance.csv
adjacent_think_utility.csv
long_distance_performance.csv
long_distance_utility.csv
intra_iteration_performance.csv
intra_iteration_utility.csv
```

These files provide the six supervised signals consumed by the downstream prediction head. The generated CSV files are local experiment artifacts and are ignored by Git.
