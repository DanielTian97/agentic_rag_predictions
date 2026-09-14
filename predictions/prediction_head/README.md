# Prediction head

Implements the camera-ready MLP prediction head for partial answer quality (`performance`) and partial utility (`utility`). The final model uses hidden layers `(16, 8)` and sequential windows `w ∈ {1, 3, 5}`.

Two paper configurations are supported:

- `("unsupervised", "supervised")` → `unsup_sup`
- `("unsupervised", "supervised", "probing")` → `unsup_sup_prob`

`fit_prediction_head(...)` can save held-out predictions under `prediction_results/` for later use by the controller. See the repository-level README for the full setup.
