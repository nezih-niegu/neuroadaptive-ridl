# Neuroadaptive RIDL

**Cognitive equivalence of VR–haptic surgical teleoperation, and a hierarchical diffusion–RL agent trained on EEG-annotated demonstrations.**

![Visual abstract](docs/visual_abstract.png)

This repository holds the code, derived results and manuscript source for the unpublished manuscript *Neuroadaptive Decision-Making AI for Immersive Surgical Robotics Training*, an extension of the MCC thesis *Neuroadaptive Decision-Making AI* (Tecnológico de Monterrey, 2025).

Participants (N = 10) teleoperated a dVRK/AMBF simulator under two interfaces, within subject:

- **VR + haptics**: Meta Quest 3 stereo view with Novint Falcon force feedback.
- **2D**: a monitor without haptics.

During each session we recorded 8-channel dry EEG, tool kinematics and endoscope video. We then asked two questions:

1. **Does immersion change how people engage?** We used a continuous Cognitive Flow Score (CFS) from an LGAF-CNN EEG decoder, movement-quality metrics, equivalence tests (TOST) and Bayes factors.
2. **Can the engagement-annotated demonstrations train an autonomous agent?** We tested **RIDL**, a two-level agent. A frozen Stable-Diffusion visuomotor skill prior learned by behavior cloning is refined by PPO on a CFS-weighted reward, across a subject-by-subject curriculum. The test is a subject-wise cross-validated ablation.

## Key results (human data)

| Measure | VR + haptics | 2D | Test |
|---|---|---|---|
| CFS (engagement) | 2.21 ± 0.25 | 2.10 ± 0.28 | d_z = 0.32, p = 0.33, BF₀₁ = 2.1; ±10% equivalence not established |
| Jerk optimality | 0.939 | 0.932 | **equivalent** within ±10% (TOST p < 0.001) |
| Smoothness (LDLJ) | −20.25 | −20.30 | **equivalent** within ±10% (TOST p < 0.001) |
| Path efficiency | 0.36 | 0.26 | d_z = 0.76, p = 0.04 uncorrected (n.s. after Holm) |
| CFS → optimality (5 s windows) | | | β = 0.11, p = 0.007 (random intercept); p = 0.12 with random slopes |

Immersion produced no measurable cognitive or motor cost, and moment-to-moment engagement tracked movement optimality in both interfaces. VR always preceded 2D, so condition order was not counterbalanced; treat this as a pilot. Full numbers are written to `results/ablation_summary.csv` and `paper/results_macros.tex` by `run_all.sh`.

## Key results (RIDL ablation)

5 subject folds × 2 demonstration sources × 2 seeds = 20 runs per variant; error = held-out 3D position error of the 1 s displacement (mean ± 95% CI); Δ = variant − RIDL, Wilcoxon signed-rank.

| Variant | Error (mm) | Δ vs RIDL |
|---|---|---|
| RIDL (BC → PPO, full) | 13.79 ± 3.18 | |
| BC only | 10.68 ± 3.92 | −3.11 (p = 0.001) |
| PPO from scratch | 15.33 ± 1.93 | +1.54 (p = 0.26) |
| w/o SD-UNet (no vision) | 8.42 ± 1.14 | −5.37 (p < 0.001) |
| Raw pixels (no SD prior) | 15.00 ± 2.00 | +1.22 (p = 0.11) |
| w/o CFS-weighted reward | 13.68 ± 1.53 | −0.11 (p = 0.45) |
| w/o CFS state input | 13.63 ± 3.44 | −0.16 (p = 0.52) |
| Pooled (no curriculum) | 14.05 ± 1.97 | +0.26 (p = 0.43) |
| Constant-velocity baseline | 11.85 ± 2.57 | −1.93 (p = 0.77) |
| Hold-position baseline | 4.93 ± 0.98 | −8.85 (p = 0.002) |

This is a negative result. In the demonstration-replay setting, where actions change the reward but not the next state, no learned policy beat holding the current pose. PPO degraded the imitation prior, the frozen SD features did not beat raw pixels or no vision, and the CFS reward weight and CFS input had no detectable effect. Agents trained on VR and 2D demonstrations did not differ (12.19 vs 15.39 mm, p = 0.35), but equivalence was not established. Testing a CFS-weighted reward properly needs a closed-loop simulator.

## Repository layout

```
ridl/            RIDLPolicy network and the original thesis training utilities
                 (model, dataset, Gymnasium env, PPO rollout buffer, DH kinematics)
experiments/     Frontiers analyses: every number in the paper comes from here
  common.py                 data loading (thesis conventions)
  human_analysis.py         CFS + kinematics: paired tests, TOST, Bayes factors, mixed models
  extract_unet_features.py  caches the frozen SD-UNet features (one pass over all frames)
  ridl_ablation.py          hierarchical RIDL + ablations + learning curves
  figures_human.py          RGB flow/optimality trajectories, paired plots, CFS–quality
  figures_rl.py             vis2-style trajectory panels + GIF, ablation bars, cross-domain matrix
  visual_abstract.py        docs/visual_abstract.png
  eval_thesis_ckpts.py      re-assesses the thesis checkpoints on never-seen subjects 8-9
  make_macros.py            paper/results_macros.tex + LaTeX tables (no hand-typed numbers)
  run_all.sh                end-to-end reproduction
thesis/          legacy real-time PPO-vs-RIDL viewers (see thesis/README.md for known issues)
notebooks/       preprocessing / LGAF, RGB trajectory encoding, vis2 animation (outputs stripped)
paper/           Frontiers LaTeX source (manuscript.tex, generated macros and tables)
figures/         rendered figures (PNG + PDF)
results/         derived, anonymized results (CSV / JSON)
```

## Setup

The project is managed with [uv](https://docs.astral.sh/uv/) (Python 3.12, locked in `uv.lock`):

```bash
uv sync                          # core: analyses, RIDL, figures
uv sync --extra viewer           # + VisPy/GLFW for the legacy thesis viewers
uv sync --extra notebooks        # + JupyterLab for notebooks/
```

CPU is sufficient; a GPU only speeds up feature extraction. The Stable-Diffusion v1.5 UNet is downloaded from the Hugging Face hub on first use. If a thesis checkpoint is placed at `checkpoints/ridl_policy_ppo_ds1.pth`, its frozen UNet weights are used instead.

## Data

Raw EEG, video and kinematics are **not distributed** (participant privacy). The code expects them under `data/VR-NVR/`, or a path set in `RIDL_DATA`:

```
data/VR-NVR/
  LGAF/    subject{i}-{RH|LH}-{VR|NVR}_{Reference|Task1|Task2}.csv   (timestamp, pose, forces, Va, Vb, Task_Ratio, F)
  Stream/  subject{i}-{RH|LH}-{VR|NVR}/left_<timestamp>.png          (endoscope frames)
  NIC/     Subject-{i}/*.easy|*.info                                  (Enobio EEG, 500 Hz)
```

The derived, anonymized results used in the paper are in `results/`.

## Reproduce

```bash
cd experiments
THREADS=16 ./run_all.sh          # ~10 h on a 22-core CPU, mostly the UNet feature pass
```

Individual steps:

```bash
uv run python human_analysis.py && uv run python figures_human.py   # human-data statistics and figures (minutes)
uv run python extract_unet_features.py                              # frozen SD-UNet cache
uv run python ridl_ablation.py --seeds 2                            # ablation grid
uv run python ridl_ablation.py --curves                             # sample-efficiency curves
uv run python eval_thesis_ckpts.py --ckpt-root <thesis folder>      # thesis checkpoints, held-out subjects 8-9
uv run python figures_rl.py && uv run python make_macros.py && uv run python visual_abstract.py
```

Compile the paper with `paper/build.sh` (pdfLaTeX + BibTeX, as on Overleaf; needs `texlive-latex-extra texlive-science texlive-fonts-recommended`). Figures are read from `../figures/`.

## Differences from the thesis implementation

The Frontiers analyses keep the thesis architecture (RIDLPolicy: frozen SD-UNet, Conv1D + 4-layer Transformer, additive fusion, Gaussian policy and value heads) and the two-level BC → PPO training with a subject curriculum. They change the following:

- **Action-dependent reward.** The thesis environment returned the logged reward whatever the action; the reward is now a CFS-weighted tracking term plus a jerk term.
- **Prediction error.** Evaluation measures true prediction error on held-out subjects (5-fold subject-wise CV) against hold-position and constant-velocity baselines. Normalization statistics come from training subjects only.
- **Policy inputs and target.** The numeric encoder receives the kinematic history, as described in the paper's Methods, and the policy predicts 1 s action chunks.
- **Feature caching.** Frozen-UNet features are cached. This is numerically identical to recomputing them and makes the ablation tractable on CPU.

See `thesis/README.md` for the issues found in the original evaluation script.

## License

Code: [MIT](LICENSE). The manuscript text and figures are not covered by this license. Raw participant data are not distributed.
