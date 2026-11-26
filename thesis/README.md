# Thesis real-time comparison scripts (legacy)

These are the original MCC-thesis scripts that replay a recorded session and
compare a PPO checkpoint with a RIDL checkpoint in real time (VisPy 3D view,
OpenCV filter windows, Kalman residuals). They are kept for provenance and
for the live demo. **Do not use them for reported numbers.**

Known issues that the Frontiers analyses in `../experiments/` correct:

| Issue | Where | Effect | Fix in `experiments/` |
|---|---|---|---|
| "Error" is the Kalman innovation of each trajectory against itself, not prediction vs. ground truth | `realtime_compare_vispy.py` (`kf_r.update(pred_p)`) | thesis Table 6.5 does not measure accuracy | `ridl_ablation.evaluate`: ‖x̂ − x‖₂ in mm, then Kalman-smoothed |
| RIDL history stores the PPO prediction | `hist_rr.append(pred_p)` | the RIDL and PPO curves are identical | separate predictions per model |
| Normalized absolute output added to current pose as a delta | `pred_p = cur_p + dp[:3]` | units mismatch | explicit Δ-pose target, de-normalized |
| Dataset normalization fitted on the test segment | `ConvImageDiffusionDataset(df_test)` | test-set leakage | statistics from training subjects only |
| PPO reward independent of the action | `ridl/env.py` `step()` | RL gradient carries no signal | action-dependent, CFS-weighted reward |

Usage (requires the raw data and thesis checkpoints, which are not distributed):

```bash
python thesis/realtime_compare_vispy.py --csv <session.csv> --image-dir <frames/> \
    --ppo-model <ppo.pth> --ridl-model <ridl.pth> --device cpu
```
