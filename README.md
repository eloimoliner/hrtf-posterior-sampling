# HRTF estimation using a score-based prior

Code for E. Thuillier, J.-M. Lemercier, E. Moliner, T. Gerkmann, V. Välimäki, "HRTF Estimation using a Score-based Prior",
in *Proc. IEEE Int. Conf. Acoustics, Speech and Signal Processing (ICASSP)*, 2025, pp. 1-5. [arXiv:2410.01562](https://arxiv.org/abs/2410.01562)

This repository holds the code for the experiment of Table I (44.1 kHz).

## Contents
| Path | What |
|---|---|
| `checkpoints/` | the DoA-conditioned HRTF prior used in the paper (1D U-Net, 752k parameters, HUTUBS, 44.1 kHz) |
| `downstream_tasks/hrtf_estimation.py` | the method: posterior sampling of the HRTF + BRIR model optimisation |
| `downstream_tasks/hrtf_estimation.yaml` | its configuration (T=101 steps, zeta=1.25, ...) |
| `downstream_tasks/brir_model.py`, `initialization.py` | BRIR model and its initialisation |
| `downstream_tasks/baselines.py` | Random / Generic / Nearest Neighbour baselines (needs HUTUBS) |
| `data/test_small/` | 12 test samples, with precomputed baselines |
| `scripts/run_test.sh` | runs the method on a test set (parallel, CPU and/or GPUs) |
| `scripts/evaluate.py` | Table I and the error-vs-azimuth plot |
| `train.py`, `training/` | training of the prior |

## Environment
Managed with [uv](https://docs.astral.sh/uv/) (Python 3.9, PyTorch 2.6 + CUDA 12.4, versions pinned in `uv.lock`):
```bash
uv sync                 # creates .venv
uv run python ...       # or source .venv/bin/activate
```

## Data
`data/test_small/` holds 12 examples of the paper's test set (2 per test subject), with the baselines already computed.
The prior was trained on the [HUTUBS](https://depositonce.tu-berlin.de/handle/11303/9429) HRTF database, which is
only needed to retrain it or to recompute the baselines.

## Running
```bash
# proposed method (~5-10 min per sample; samples run in parallel; resumable). GPUS="" runs on CPU.
GPUS="0" scripts/run_test.sh data/test_small 12 4

# evaluation -> data/test_small/evaluation/{table1.md, error_vs_azimuth.png, per_sample.csv}
uv run python scripts/evaluate.py data/test_small
```

## Training the prior
Needs the HUTUBS simulated HRIRs as pointwise (DoA, HRIR) pairs at 44.1 kHz, listed in `{train,valid,test}_entries.json`.
```bash
uv run python train.py dataset_dir=<pointwise_hrir_44p1kHz/hutubs_simulated>   # checkpoints in experiments/score_model
```

## Expected results
LRE / LMD in dB, mean ± std over samples. `data/test_small` results are from `scripts/run_test.sh`.

| Method | `data/test_small` (12 samples) | Full test set (599 samples) | Paper (Table I) |
|---|---|---|---|
| Random | -3.41 ± 2.03 / 4.62 ± 0.86 | -3.59 ± 1.84 / 4.92 ± 0.82 | -3.43 ± 1.59 / 4.58 ± 0.69 |
| Generic | -3.90 ± 1.37 / 4.85 ± 0.84 | -4.01 ± 1.75 / 4.77 ± 0.74 | -4.01 ± 1.75 / 4.77 ± 0.74 |
| Proposed | -9.47 ± 5.63 / 2.18 ± 0.90 | -8.48 ± 5.21 / 2.68 ± 1.23 | -9.20 ± 5.64 / 2.28 ± 1.11 |
| Nearest Neighbour | -7.51 ± 1.35 / 3.68 ± 0.52 | -7.50 ± 1.46 / 3.76 ± 0.61 | -7.50 ± 1.46 / 3.76 ± 0.61 |

 The full-set Proposed result differs slightly from the paper because of different test speech utterances and hardware/library versions. 

## Citation
```bibtex
@inproceedings{thuillier2025hrtf,
  title={HRTF Estimation using a Score-based Prior},
  author={Thuillier, Etienne and Lemercier, Jean-Marie and Moliner, Eloi and Gerkmann, Timo and V{\"a}lim{\"a}ki, Vesa},
  booktitle={ICASSP 2025 - 2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={1--5},
  year={2025},
  month={apr},
  organization={IEEE}
}
```
