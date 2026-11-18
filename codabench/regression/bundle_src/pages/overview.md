# Smart Pixels regression

From one silicon-pixel cluster, as a 16 × 16 pixel readout chip sees it, estimate where the particle crossed the
sensor mid-plane (x, y in µm) and at which angles (cot α, cot β). Probabilistic models also give a 1σ
uncertainty for each. Measured in one layer, these let the detector select or seed tracks at the sensor edge
instead of shipping every pixel.

There are two tracks:
- **asic**: inputs at time slices 0 and 19 only, each pixel digitized to 2 bits by three strictly increasing
  thresholds you declare, at most 2734 parameters, and weights and activations of at most 8 bits, all declared in
  `descriptor.json`;
- **open**: no constraints.

The reference models and the full protocol are in the benchmark card (`benchmarks/regression/README.md` in the
benchmark repository, https://github.com/lgray/smartpixels-benchmark). Das et al., arXiv:2602.15946, define the
task; the dataset is Zenodo 10.5281/zenodo.18472791 (CC-BY-4.0).
