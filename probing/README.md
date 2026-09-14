# Probing

Intermediate-answer probing for Search-R1 and R1-Searcher. The released implementation records the forced intermediate answer together with token log-probabilities and the scalar confidence used by the paper.

Probe 0 is the zero-retrieval answer obtained **after initial reasoning `r0` and before external retrieval**. Iteration row `i` uses `probe_{i+1}` and the confidence change from `probe_i`.

## CLI

The live probing path follows the paper's canonical E5 dense-retrieval setup. The supplied dense index should be a prepared PyTerrier-DR FlexIndex backed by the HNSW FAISS index used in the experiments.

```bash
python -m probing.run_probing run \
    --queries-csv <QUERY_CSV> \
    --model search_r1 \
    --dense-index <E5_INDEX> \
    --output-csv <PROBING_RESULTS_CSV> \
    --features-output-csv <PROBING_FEATURES_CSV> \
    --top-k 3
```

Use `--model r1_searcher` for R1-Searcher. `--hf-model` can override the default Hugging Face model used by the selected agent. Additional backend or agent constructor settings can be supplied as JSON through `--backend-args` and `--agent-kwargs`.

The trajectory-level output contains the probe answers and confidence traces. When `--features-output-csv` is supplied, the runner also writes the aligned iteration-level `prob` and `diff_prob` features used by the probing-enhanced prediction setting.
