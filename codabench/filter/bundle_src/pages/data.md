# Data

Download the public dataset from Zenodo (10.5281/zenodo.10783560, version 3). The benchmark repository's
`scripts/fetch_data.py` fetches it and `smartpix_bench.filter.cache` builds the file-disjoint train and test
splits and their features. The test set is 32 of the 160 files: 1,599,888 clusters.
`starting_kit/test_sample_ids.csv` lists their ids. The field definitions and forbidden inputs are in
`data/SCHEMA.md`.

Train on the training split only. Do not augment or add data, and do not fit, tune or select on the test labels.
