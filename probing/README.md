# Probing

Intermediate-answer probing for Search-R1 and R1-Searcher. The released implementation records the forced intermediate answer together with token log-probabilities and the scalar confidence used by the paper.

Probe 0 is the zero-retrieval answer obtained **after initial reasoning `r0` and before external retrieval**. Iteration row `i` uses `probe_{i+1}` and the confidence change from `probe_i`.
