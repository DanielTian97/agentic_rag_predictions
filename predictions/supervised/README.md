# Supervised predictors

Three relation-specific regressors provide supervised signals for the final prediction head:

- `adjacent_think` (AT)
- `long_distance` (LD)
- `intra_iteration` (II)

Each is trained for both partial utility `U_i` and partial answer quality `P_i`, producing six continuous signals. `prediction_results/` is reserved for generated prediction CSVs. See the repository-level README for the full experimental setup.
