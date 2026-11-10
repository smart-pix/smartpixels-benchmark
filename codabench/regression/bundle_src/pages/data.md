# Data

Download the public dataset from Zenodo (10.5281/zenodo.18472791). The benchmark repository's
`scripts/fetch_data.py` fetches it. Train and validate on the 3sExtended set (file-disjoint 80 / 20, seed 0, in
`smartpix_bench.regression.data`). The test sets are the contained clusters of 3s (1,377,888) and 2s
(1,401,971). A cluster is contained when it has less than 50 e on the array's border pixels at slice 19.
`starting_kit/<split>_sample_ids.csv` lists the ids. The field definitions and forbidden inputs are in
`data/SCHEMA.md`.

Do not add data, and do not fit, tune or select on the test labels. Training-time input noise is permitted.
