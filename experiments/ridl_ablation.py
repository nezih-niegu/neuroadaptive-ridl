"""RIDL ablation study (Frontiers extension of the MCC thesis).

Model: identical RIDLPolicy trainable stack (Conv1d embed + sinusoidal PE +
4-layer/8-head/ff256 Transformer, Linear(16384->512) UNet projection, additive
fusion, GELU MLP, Gaussian policy head + value head). The frozen SD-UNet output
is read from cache (extract_unet_features.py) instead of recomputed.

Hierarchical RIDL = two levels, as in the thesis:
  level 1 (skill prior): behaviour cloning on a subject's demonstrations,
  level 2 (refinement):  PPO fine-tuning on the flow-weighted reward,
  repeated over training subjects in sequence, each initialised from the
  previous subject's weights (continual transfer, thesis Fig. 4.5).

Environment fix: the thesis env returned the logged reward irrespective of the
action. Here the reward depends on the action:
  r_t = A * w(F_t) * exp(-||a_t - a*_t||^2 / (2 sigma^2 d)) + B * exp(-||a_t - a_{t-1}||^2 / (2 d))
where a*_t is the operator's next pose (normalised), w(F) is the CFS weight
(F relative to the segment mean, clipped to [0,2] and halved; =1 when flow
weighting is ablated) and the second term is the jerk/optimality bonus.

Evaluation (fixes the thesis test): position error ||x_pred - x_true||_2 in mm
on held-out subjects, Kalman-smoothed (E_traj) and its volatility (sigma_r).
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

from common import ACT, CACHE, RESULTS, SUBJECTS, merged_segment, seg_name
from ridl.model import PositionalEncoding

TASKS = ["Task1", "Task2"]
HORIZON = int(os.environ.get("RIDL_H", "10"))  # action-chunk horizon in 0.1 s steps (10 = 1 s)
TARGET = os.environ.get("RIDL_TARGET", "delta")  # "delta" (thesis test: pred = cur + dp) or "abs"
NL = 8  # numeric sequence length (thesis state_dim_per_channel)
A_W, B_W, SIGMA = 1.0, 0.1, 0.5


# ----------------------------------------------------------------------------- data
class Seg:
    def __init__(self, s, c, g, vis):
        m = merged_segment(s, c, g)
        if vis == "unet":
            z = np.load(os.path.join(CACHE, seg_name(s, c, g) + ".npz"))
            assert np.allclose(z["timestamp"], m["timestamp"].to_numpy()), seg_name(s, c, g)
        self.s, self.c, self.g = s, c, g
        self.flow = m["Flow"].to_numpy(np.float32)
        self.opt = m["Optimality"].to_numpy(np.float32)
        self.act = m[ACT].to_numpy(np.float32)
        if vis == "unet":
            self.img = z["feats"]
        elif vis == "pixels":
            self.img = pixels(m["image_path"].tolist(), s, c, g)
        else:
            self.img = np.zeros((len(m), 1), np.float16)
        w = np.clip(self.flow / self.flow.mean(), 0, 2) / 2
        self.w = w.astype(np.float32)
        self.n = len(m) - HORIZON  # transitions t -> t+H


def pixels(paths, s, c, g):
    import cv2
    f = os.path.join(CACHE, "..", "pix64", seg_name(s, c, g) + ".npy")
    if os.path.exists(f):
        return np.load(f)
    os.makedirs(os.path.dirname(f), exist_ok=True)
    X = np.stack([cv2.resize(cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB), (64, 64),
                             interpolation=cv2.INTER_AREA).reshape(-1) for p in paths])
    X = (X.astype(np.float32) / 127.5 - 1).astype(np.float16)
    np.save(f, X)
    return X


class Norm:
    def __init__(self, segs):
        A = np.concatenate([x.act for x in segs])
        S = np.concatenate([numeric_raw(x) for x in segs])
        I = np.concatenate([x.img for x in segs]).astype(np.float32)
        if TARGET == "delta":
            A = np.concatenate([x.act[HORIZON:] - x.act[:-HORIZON] for x in segs])
        self.am, self.asd = A.mean(0), A.std(0) + 1e-6
        self.sm, self.ssd = S.mean(0), S.std(0) + 1e-6
        self.im, self.isd = I.mean(0), I.std() + 1e-6  # scalar scale: per-dim z-scoring of 16k dims overfits


def numeric_raw(seg):
    """Numeric encoder channels (paper Methods: 'positions, forces, orientations'):
    Flow, Optimality and the 9 last-step displacements a_t - a_{t-1}."""
    d = np.vstack([np.zeros((1, seg.act.shape[1]), np.float32), np.diff(seg.act, axis=0)])
    return np.concatenate([np.stack([seg.flow, seg.opt], 1), d], 1)


NC = 11


def tensors(seg, nz, use_flow=True):
    """obs_t (numeric window [2,NL], image feat), target a*_{t} = pose_{t+1}."""
    S = (numeric_raw(seg) - nz.sm) / nz.ssd
    if not use_flow:
        S[:, 0] = 0.0
    pad = np.concatenate([np.repeat(S[:1], NL - 1, 0), S])
    win = np.stack([pad[t:t + NL].T for t in range(len(S))])  # [N,NC,NL]
    img = (seg.img.astype(np.float32) - nz.im) / nz.isd
    n = seg.n
    if TARGET == "delta":
        tgt = (seg.act[HORIZON:] - seg.act[:-HORIZON] - nz.am) / nz.asd   # a_{t+H} - a_t
    else:
        tgt = ((seg.act - nz.am) / nz.asd)[HORIZON:n + HORIZON]
    return (torch.from_numpy(win[:n]).float(), torch.from_numpy(img[:n]).float(),
            torch.from_numpy(tgt).float(), torch.from_numpy(seg.w[:n]))


# ----------------------------------------------------------------------------- model
class RIDLHead(nn.Module):
    """RIDLPolicy (model.py) with the frozen UNet factored out."""

    def __init__(self, img_dim, nc=NC, act_dim=9, hid=512, layers=4, heads=8, ff=256, drop=0.1, vision=True):
        super().__init__()
        self.vision = vision
        self.num_embed = nn.Conv1d(nc, hid, 1)
        self.pos_encoder = PositionalEncoding(hid)
        el = nn.TransformerEncoderLayer(hid, heads, ff, drop, batch_first=True, activation="gelu")
        self.transformer_encoder = nn.TransformerEncoder(el, layers)
        self.img_drop = nn.Dropout(0.3)
        self.unet_projection = nn.Linear(img_dim, hid)
        self.projection = nn.Sequential(nn.Linear(hid, hid // 2), nn.GELU(), nn.Dropout(drop))
        self.mean_head = nn.Linear(hid // 2, act_dim)
        self.value_head = nn.Linear(hid // 2, 1)
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.5))

    def forward(self, num, img):
        x = self.pos_encoder(self.num_embed(num).transpose(1, 2))
        x = self.transformer_encoder(x).mean(1)
        if self.vision:
            x = x + self.unet_projection(self.img_drop(img))
        z = self.projection(x)
        mu = self.mean_head(z)
        return mu, self.log_std.exp().expand_as(mu), self.value_head(z).squeeze(-1)


# ----------------------------------------------------------------------------- training
def bc(model, data, epochs, lr=3e-4, bs=64):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    num, img, tgt, _ = data
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(len(tgt))
        for i in range(0, len(tgt), bs):
            j = perm[i:i + bs]
            mu, _, _ = model(num[j], img[j])
            loss = ((mu - tgt[j]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()


def rollout(model, data, flow_w):
    num, img, tgt, w = data
    model.eval()
    with torch.no_grad():
        mu, sd, v = model(num, img)
        dist = torch.distributions.Normal(mu, sd)
        a = dist.sample()
        lp = dist.log_prob(a).sum(-1)
    d = a.shape[1]
    track = torch.exp(-((a - tgt) ** 2).sum(-1) / (2 * SIGMA ** 2 * d))
    prev = torch.cat([a[:1], a[:-1]])
    smooth = torch.exp(-((a - prev) ** 2).sum(-1) / (2 * d))
    ww = w if flow_w else torch.ones_like(w)
    r = A_W * ww * track + B_W * smooth
    return a, lp, v, r


def gae(r, v, gamma=0.99, lam=0.95):
    adv = torch.zeros_like(r)
    last = 0.0
    for t in reversed(range(len(r))):
        nv = v[t + 1] if t + 1 < len(r) else 0.0
        delta = r[t] + gamma * nv - v[t]
        last = delta + gamma * lam * last
        adv[t] = last
    return adv, adv + v


def ppo(model, datas, iters, flow_w, rng, lr=1e-4, clip=0.2, epochs=4, bs=64,
        ent=1e-3, curve=None, evalf=None, step0=0):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    steps = step0
    for it in range(iters):
        data = datas[rng.integers(len(datas))]
        num, img, _, _ = data
        a, lp_old, v_old, r = rollout(model, data, flow_w)
        adv, ret = gae(r, v_old)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        model.train()
        for _ in range(epochs):
            perm = torch.randperm(len(r))
            for i in range(0, len(r), bs):
                j = perm[i:i + bs]
                mu, sd, v = model(num[j], img[j])
                dist = torch.distributions.Normal(mu, sd)
                ratio = (dist.log_prob(a[j]).sum(-1) - lp_old[j]).exp()
                pl = -torch.min(ratio * adv[j], ratio.clamp(1 - clip, 1 + clip) * adv[j]).mean()
                vl = ((v - ret[j]) ** 2).mean()
                loss = pl + 0.5 * vl - ent * dist.entropy().sum(-1).mean()
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5); opt.step()
        steps += len(r)
        if curve is not None:
            curve.append(dict(steps=steps, reward=float(r.mean()), **(evalf() if evalf else {})))
    return steps


# ----------------------------------------------------------------------------- evaluation
def kalman(z, q=1e-5, r=0.1):
    x, P, out = z[0], 1.0, []
    for v in z:
        P += q
        K = P / (P + r)
        x += K * (v - x)
        P *= (1 - K)
        out.append(x)
    return np.array(out)


def evaluate(model, tests, nz, use_flow, keep=False):
    if model is not None and model != "cv":
        model.eval()
    rows, trajs = [], {}
    with torch.no_grad():
        for seg in tests:
            num, img, tgt, _ = tensors(seg, nz, use_flow)
            if model == "cv":  # constant-velocity baseline
                last = HORIZON * np.vstack([np.zeros((1, 9)), np.diff(seg.act, axis=0)])[:seg.n]
                mu = (last - nz.am) / nz.asd
            elif model is not None:
                mu = model(num, img)[0].numpy()
            else:  # trivial baseline: zero displacement (delta) / training-mean pose (abs)
                mu = ((0 - nz.am) / nz.asd) * np.ones_like(tgt.numpy()) if TARGET == "delta" else np.zeros_like(tgt.numpy())
            pred = mu * nz.asd + nz.am
            if TARGET == "delta":
                pred = seg.act[:seg.n] + pred
            true = seg.act[HORIZON:seg.n + HORIZON]
            e = np.linalg.norm(pred[:, :3] - true[:, :3], axis=1) * 1000  # mm
            ef = kalman(e)
            jerk = np.abs(np.diff(pred[:, :3], 3, axis=0)).sum(1).mean() * 1000
            rows.append(dict(subject=seg.s, cond=seg.c, seg=seg.g, err_mm=e.mean(),
                             E_traj=ef.mean(), sigma_r=np.sqrt(((ef - ef.mean()) ** 2).mean()),
                             nmse=float(((mu - tgt.numpy()) ** 2).mean()), pred_jerk_mm=jerk))
            if keep:
                trajs[seg_name(seg.s, seg.c, seg.g)] = dict(true=true[:, :3], pred=pred[:, :3],
                                                            flow=seg.flow[:seg.n], err=e)
    return rows, trajs


# ----------------------------------------------------------------------------- variants
VARIANTS = {
    # name: (vision, bc_epochs, ppo_iters, flow_reward, flow_input, schedule)
    "RIDL (full)":        ("unet", 10, 6, True, True, "sequential"),
    "PPO from scratch":   ("unet", 0, 6, True, True, "sequential"),
    "BC only":            ("unet", 10, 0, True, True, "sequential"),
    "w/o SD-UNet (no vision)": ("none", 10, 6, True, True, "sequential"),
    "Raw pixels (no SD prior)": ("pixels", 10, 6, True, True, "sequential"),
    "w/o flow-weighted reward": ("unet", 10, 6, False, True, "sequential"),
    "w/o CFS state input": ("unet", 10, 6, True, False, "sequential"),
    "Pooled (no curriculum)": ("unet", 10, 6, True, True, "pooled"),
}


def run(variant, source, fold, seed, folds, cache, curve_every=False):
    vision, bce, ppi, flow_w, flow_in, sched = VARIANTS[variant]
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    test_s = folds[fold]
    train_s = [s for s in SUBJECTS if s not in test_s]
    key = lambda s, c, g: (vision, s, c, g)
    for s in SUBJECTS:
        for c in ("VR", "NVR"):
            for g in TASKS:
                if key(s, c, g) not in cache:
                    cache[key(s, c, g)] = Seg(s, c, g, vision)
    train = [cache[key(s, source, g)] for s in train_s for g in TASKS]
    tests = [cache[key(s, c, g)] for s in test_s for c in ("VR", "NVR") for g in TASKS]
    nz = Norm(train)
    img_dim = train[0].img.shape[1]
    model = RIDLHead(img_dim, vision=vision != "none")
    curve = [] if curve_every else None
    evalf = (lambda: dict(test_nmse=np.mean([r["nmse"] for r in evaluate(model, [t for t in tests if t.c == source], nz, flow_in)[0]]))) if curve_every else None
    steps = 0
    t0 = time.time()
    if sched == "sequential":
        for s in train_s:  # hierarchical curriculum over subjects
            datas = [tensors(cache[key(s, source, g)], nz, flow_in) for g in TASKS]
            if bce:
                bc(model, tuple(torch.cat(x) for x in zip(*datas)), bce)
                steps += 0  # BC uses logged demos, no environment interaction
            if ppi:
                steps = ppo(model, datas, ppi, flow_w, rng, curve=curve, evalf=evalf, step0=steps)
    else:
        datas = [tensors(x, nz, flow_in) for x in train]
        if bce:
            bc(model, tuple(torch.cat(x) for x in zip(*datas)), bce)
        if ppi:
            steps = ppo(model, datas, ppi * len(train_s), flow_w, rng, curve=curve, evalf=evalf)
    rows, trajs = evaluate(model, tests, nz, flow_in, keep=(seed == 0))
    for r in rows:
        r.update(variant=variant, source=source, fold=fold, seed=seed, env_steps=steps,
                 train_sec=time.time() - t0)
    return rows, trajs, curve, (model, nz)


def curves(seeds, iters, threads):
    """Sample efficiency: held-out NMSE after every PPO update, RIDL vs PPO-scratch."""
    torch.set_num_threads(threads)
    folds = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
    cache, C = {}, {}
    for name, v in (("RIDL (BC → PPO)", "RIDL (full)"), ("PPO from scratch", "PPO from scratch")):
        vis, bce, _, fw, fi, sch = VARIANTS[v]
        VARIANTS["_curve"] = (vis, bce, iters, fw, fi, sch)
        C[name] = []
        for f in range(len(folds)):
            for sd in range(seeds):
                _, _, cv, _ = run("_curve", "VR", f, sd, folds, cache, curve_every=True)
                C[name].append(cv)
                print(name, f, sd, cv[-1], flush=True)
    json.dump(C, open(os.path.join(RESULTS, "curves.json"), "w"))
    # steps for PPO-scratch to reach the error RIDL has after its first PPO update
    out = {}
    r0 = np.array([c[0]["test_nmse"] for c in C["RIDL (BC → PPO)"]])
    rf = np.array([c[-1]["test_nmse"] for c in C["RIDL (BC → PPO)"]])
    pf = np.array([c[-1]["test_nmse"] for c in C["PPO from scratch"]])
    reach = []
    for c, thr in zip(C["PPO from scratch"], rf):
        hit = [p["steps"] for p in c if p["test_nmse"] <= thr]
        reach.append(hit[0] if hit else np.nan)
    out = dict(ridl_after_first_update=r0.mean(), ridl_final=rf.mean(), ppo_final=pf.mean(),
               ppo_steps_to_ridl_final=float(np.nanmean(reach)) if np.isfinite(reach).any() else None,
               ppo_reached_frac=float(np.isfinite(reach).mean()),
               total_steps=C["RIDL (BC → PPO)"][0][-1]["steps"])
    json.dump(out, open(os.path.join(RESULTS, "curves_summary.json"), "w"), indent=1, default=float)
    print(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--curves", action="store_true")
    ap.add_argument("--curve-iters", type=int, default=25)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS))
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--tag", default="ablation")
    a = ap.parse_args()
    if a.curves:
        return curves(a.seeds, a.curve_iters, a.threads)
    torch.set_num_threads(a.threads)
    folds = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
    out_csv = os.path.join(RESULTS, f"{a.tag}.csv")
    done = set()
    allrows = []
    if os.path.exists(out_csv):
        import pandas as pd
        prev = pd.read_csv(out_csv)
        allrows = prev.to_dict("records")
        done = set(zip(prev.variant, prev.source, prev.fold, prev.seed))
    cache = {}
    traj_dir = os.path.join(RESULTS, "trajectories")
    os.makedirs(traj_dir, exist_ok=True)
    for v in a.variants:
        for src in ("VR", "NVR"):
            for f in range(len(folds)):
                for sd in range(a.seeds):
                    if (v, src, f, sd) in done:
                        continue
                    rows, trajs, _, _ = run(v, src, f, sd, folds, cache)
                    allrows += rows
                    import pandas as pd
                    pd.DataFrame(allrows).to_csv(out_csv, index=False)
                    if trajs:
                        np.savez_compressed(os.path.join(traj_dir, f"{v.replace('/', '_').replace(' ', '_')}__{src}__f{f}.npz"),
                                            **{k + "__" + kk: vv for k, d in trajs.items() for kk, vv in d.items()})
                    e = np.mean([r["err_mm"] for r in rows])
                    print(f"{v:28s} src={src:3s} fold={f} seed={sd} err={e:.2f}mm t={rows[0]['train_sec']:.0f}s", flush=True)
    # trivial baseline: predict training-mean pose
    for src in ("VR", "NVR"):
        for f in range(len(folds)):
            vk = next(k[0] for k in cache)
            train = [cache[(vk, s, src, g)] for s in SUBJECTS if s not in folds[f] for g in TASKS]
            tests = [cache[(vk, s, c, g)] for s in folds[f] for c in ("VR", "NVR") for g in TASKS]
            for mdl, name in ((None, "Hold-position baseline" if TARGET == "delta" else "Mean-pose baseline"),
                              ("cv", "Constant-velocity baseline")):
                rows, _ = evaluate(mdl, tests, Norm(train), True)
                for r in rows:
                    r.update(variant=name, source=src, fold=f, seed=0, env_steps=0, train_sec=0)
                allrows += rows
    import pandas as pd
    pd.DataFrame(allrows).drop_duplicates(["variant", "source", "fold", "seed", "subject", "cond", "seg"]).to_csv(out_csv, index=False)


if __name__ == "__main__":
    main()
