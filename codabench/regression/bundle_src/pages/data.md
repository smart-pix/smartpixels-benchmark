# Data

Download the public dataset from Zenodo (10.5281/zenodo.18472791). The benchmark repository's
`scripts/fetch_data.py` fetches it. Train and validate on the 3sExtended set (file-disjoint 80 / 20, seed 0, in
`smartpix_bench.regression.data`). The Codabench test set is the 1,377,888 contained clusters of 3s. The second
test set, 2s (1,401,971), is scored only by the benchmark repository, to keep uploads small. A cluster is contained
when it has less than 50 e on the array's border pixels at slice 19. `starting_kit/3s_sample_ids.csv` lists the
ids. The field definitions and forbidden inputs are in `data/SCHEMA.md`.

Do not add data, and do not fit, tune or select on the test labels. Training-time input noise is permitted.
