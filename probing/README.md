# Probing

Intermediate-answer probing for Search-R1 and R1-Searcher. The released implementation records the forced intermediate answer together with token log-probabilities and the scalar confidence used by the paper.

Probe 0 is the zero-retrieval answer obtained **after initial reasoning `r0` and before external retrieval**. Iteration row `i` uses `probe_{i+1}` and the confidence change from `probe_i`.

## CLI

A live probing run requires a PyTerrier retriever. Because index locations and construction differ across installations, the CLI accepts a small Python retriever factory in `module:function` form. The factory must return a `pyterrier.Transformer`.

For example, if `my_retriever.py` provides `build_retriever(...)`, run:

```bash
python -m probing.run_probing run \
    --queries-csv <QUERY_CSV> \
    --model search_r1 \
    --retriever-factory my_retriever:build_retriever \
    --retriever-kwargs '{"index_path": "<E5_INDEX>"}' \
    --output-csv <PROBING_RESULTS_CSV> \
    --features-output-csv <PROBING_FEATURES_CSV> \
    --top-k 3
```

Use `--model r1_searcher` for R1-Searcher. `--hf-model` can override the default Hugging Face model used by the selected agent. Additional backend or agent constructor settings can be supplied as JSON through `--backend-args` and `--agent-kwargs`.

The trajectory-level output contains the probe answers and confidence traces. When `--features-output-csv` is supplied, the runner also writes the aligned iteration-level `prob` and `diff_prob` features used by the probing-enhanced prediction setting.
