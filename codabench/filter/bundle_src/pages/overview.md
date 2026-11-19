# Smart Pixels filter

At the HL-LHC, pixel detectors produce more data than can leave the detector at 40 MHz. Clusters from
low-momentum tracks (|pT| ≤ 0.2 GeV) curve in the 3.8 T field, cross the 100 µm sensor at shallow angles and
are of no use to the trigger. Rejecting them inside the pixel readout chip cuts the data volume before it is read
out. The task is to classify one cluster, as a 13 × 21 pixel chip sees it, as high-pT (keep) or low-pT of either
charge (reject).

There are two tracks:
- **asic**: the inputs the ASIC provides (each pixel digitized to 2 bits at 400 / 1600 / 2400 e before the row
  sum, y0 as a 6-bit code) and at most 2819 parameters, declared in `descriptor.json`;
- **open**: no constraints.

The reference models and the full protocol are in the benchmark card (`benchmarks/filter/README.md` in the
benchmark repository, https://github.com/smart-pix/smartpixels-benchmark). Yoo et al., arXiv:2310.02474, define the
task; the dataset is Zenodo 10.5281/zenodo.10783560 v3 (CC-BY-4.0).
