"""Re-assess the thesis-trained RIDL checkpoints with the corrected metric.

The thesis trained one model per condition/task, sequentially over subjects
(ds1 = subject0, ..., ds10 = subject9), each initialised from the previous one.
Checkpoint ds8 has therefore seen subjects 0-7 only, so subjects 8 and 9 are a
genuine held-out test set for the thesis models.

Thesis conventions reproduced exactly (dataset.py / model.py):
  state  = z-scored {Flow, Optimality}, zero-padded to 16 and reshaped to [2, 8]
  action = z-scored absolute pose/force at the same time step
  image  = frozen SD-UNet(64x64) features, fed raw to unet_projection
Correction: z-score statistics come from the training subjects (0-7), not from
the test segment itself as in the thesis testing script.
Metric: 3D position error ||x_hat - x||_2 in mm; baseline = training-mean pose.
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch

from common import ACT, CACHE, RESULTS, ROOT, merged_segment, seg_name
from ridl_ablation import RIDLHead, kalman

TRAIN, TEST = list(range(8)), [8, 9]


def thesis_state(seg_df, mean, std):
    S = (seg_df[["Flow", "Optimality"]].to_numpy(np.float32) - mean) / std
    S = np.concatenate([S, np.zeros((len(S), 14), np.float32)], 1)
    return torch.from_numpy(S.reshape(len(S), 2, 8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-root", default=ROOT, help="folder holding outputs_{vr,nvr}_{task1,task2}/")
    ap.add_argument("--ds", type=int, default=8)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    rows = []
    for cond in ("VR", "NVR"):
        for task in ("Task1", "Task2"):
            tr = [merged_segment(s, cond, task) for s in TRAIN]
            S = np.concatenate([d[["Flow", "Optimality"]].to_numpy(np.float32) for d in tr])
            A = np.concatenate([d[ACT].to_numpy(np.float32) for d in tr])
            sm, ss, am, asd = S.mean(0), S.std(0) + 1e-6, A.mean(0), A.std(0) + 1e-6
            for kind in ("imit_policy", "ridl_policy_ppo"):
                f = os.path.join(a.ckpt_root, f"outputs_{cond.lower()}_{task.lower()}", f"{kind}_ds{a.ds}.pth")
                sd = torch.load(f, map_location="cpu", mmap=True, weights_only=True)
                head = {k: v for k, v in sd.items() if not k.startswith("unet.")}
                m = RIDLHead(16384, nc=2)
                missing, unexpected = m.load_state_dict(head, strict=False)
                assert not unexpected and all(k.startswith("img_drop") for k in missing), (missing, unexpected)
                m.eval()
                for s in TEST:
                    d = merged_segment(s, cond, task)
                    z = np.load(os.path.join(CACHE, seg_name(s, cond, task) + ".npz"))
                    with torch.no_grad():
                        mu = m(thesis_state(d, sm, ss), torch.from_numpy(z["feats"].astype(np.float32)))[0].numpy()
                    pred = mu * asd + am
                    true = d[ACT].to_numpy(np.float32)
                    e = np.linalg.norm(pred[:, :3] - true[:, :3], axis=1) * 1000
                    eb = np.linalg.norm(am[:3] - true[:, :3], axis=1) * 1000
                    ef = kalman(e)
                    rows.append(dict(model="Thesis BC (imitation)" if kind == "imit_policy" else "Thesis RIDL (BC→PPO)",
                                     ckpt=f"ds{a.ds}", cond=cond, task=task, subject=s, err_mm=e.mean(),
                                     E_traj=ef.mean(), sigma_r=ef.std(), baseline_mean_pose_mm=eb.mean(),
                                     nmse=float((((mu - (true - am) / asd)) ** 2).mean())))
                    print(rows[-1], flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RESULTS, "thesis_ckpts.csv"), index=False)
    print(out.groupby("model")[["err_mm", "baseline_mean_pose_mm", "E_traj", "sigma_r", "nmse"]].mean().round(3))


if __name__ == "__main__":
    main()
