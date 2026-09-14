# Prediction-head results

Put generated prediction-head CSV files here.

The default writer stores results under subdirectories for the selected feature-group configuration, for example:

```text
unsup_sup/<pipeline>/<dataset>/w3/performance.csv
unsup_sup/<pipeline>/<dataset>/w3/utility.csv
unsup_sup_prob/<pipeline>/<dataset>/w3/performance.csv
unsup_sup_prob/<pipeline>/<dataset>/w3/utility.csv
```

These CSV files are local experiment artifacts and are ignored by Git. The controller can load the corresponding performance and utility prediction files for the configuration being evaluated.
