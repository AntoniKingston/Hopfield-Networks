# Hopfield Networks: associative memory (Project 2, SN 2026/27)

A Hopfield network used as an associative memory. It is trained with Hebb's rule and Oja's rule,
with the delta / projection (pseudo-inverse) rule as a reference. It can be simulated
synchronously or asynchronously, and every convergence step can be shown as a bitmap.

## Setup

```sh
uv sync                        # creates .venv with all dependencies (incl. pytest)
uv run python scripts/fetch_cats.py   # downloads the photos for the large-bitmap experiment
```

## Usage

```python
import numpy as np
from hopfield import Hopfield
from hopfield_datasets import load_dataset

rng = np.random.default_rng(0)

d = load_dataset("animals-14x9")  # file names are WxH, so d.shape == (9, 14)
net = Hopfield(d.shape, d.patterns)  # number of neurons = rows * cols
net.train("hebb")  # "hebb" | "oja" | "delta" | "projection"

x = net.corrupt(d.patterns[0], 0.1, rng)  # flip 10 % of the pixels
result = net.run(x, mode="async", seed=0)  # or mode="sync"
net.show_history(result.history)  # slider over the steps (Jupyter)
net.plot_history(result.history)  # grid of all steps
net.save_history_gif(result.history, "steps.gif")
```

For bitmaps too large for an N x N matrix (e.g. 200x300, N = 60 000) use
`LowRankHopfield(shape, patterns, rule="hebb" | "projection")`. It has the same API and
memory O(PN).

`notebooks/demo.ipynb` is the interactive demo used during the presentation.

## Reproducing the report

```sh
uv run python scripts/run_experiments.py           # all experiments (~10 min)
uv run python scripts/run_experiments.py --only q4 # one section: q1 q3 q4 q6 q7 q8 q9
cd report && latexmk -pdf main.tex
```

All randomness comes from `numpy.random.default_rng` seeded with `experiments.SEED = 2137`
(per-experiment sub-seeds are derived from it). Raw results go to `results/`, LaTeX tables to
`report/tables/` and figures to `report/figures/`.

## Presentation

`presentation/index.html` is a 10–15 min Polish talk (reveal.js and KaTeX are bundled in
`presentation/vendor/`, so it works offline). Open it directly in a browser; press `S` for
speaker notes with timings. It includes a live JS Hopfield demo (Hebb / Oja / projection, sync /
async) and reads figures from `report/figures/`. The patterns, Oja weights and key numbers come
from `results/`; regenerate them after rerunning experiments with

```sh
uv run python scripts/export_presentation_data.py   # writes presentation/data/patterns.js
```

## Layout

* `src/hopfield.py`: the network (`Hopfield`, `LowRankHopfield`), learning rules, dynamics,
  visualization.
* `src/hopfield_datasets.py`: dataset loading, image binarization, pattern statistics.
* `src/experiments.py`: one function per question of the assignment.
* `scripts/`: experiment runner and data download.
* `tests/`: `uv run pytest`.
* `report/`: LaTeX report (Polish).
* `presentation/`: reveal.js slides (Polish) with a live demo.
