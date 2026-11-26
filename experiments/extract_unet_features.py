"""Cache frozen Stable-Diffusion UNet features for every endoscope frame.

The RIDLPolicy UNet is frozen and called with t=0 and zero text conditioning,
so its output for a frame is deterministic. Computing it once and caching it
is mathematically identical to recomputing it inside every forward pass, and
lets all ablation runs train only the learnable heads.

Preprocessing matches dataset.py + model.py exactly:
  cv2 BGR->RGB, /255, ToTensor, Resize(64,64), Normalize(0.5,0.5),
  concat zero 4th channel, UNet(t=0, enc=zeros(1,1,768)).sample, flatten.
UNet weights are taken from the thesis checkpoint so they are bit-identical.
"""
import argparse
import os
import time

import cv2
import numpy as np
import torch
from diffusers import UNet2DConditionModel
from torchvision import transforms

from common import CACHE, CONDS, ROOT, SUBJECTS, merged_segment, seg_name

ORDER = ["Task1", "Task2", "Reference"]


def load_unet(ckpt):
    unet = UNet2DConditionModel.from_pretrained(
        "runwayml/stable-diffusion-v1-5", subfolder="unet").eval()
    if os.path.exists(ckpt):
        sd = torch.load(ckpt, map_location="cpu", mmap=True, weights_only=True)
        unet.load_state_dict({k[5:]: v for k, v in sd.items() if k.startswith("unet.")}, strict=True)
    for p in unet.parameters():
        p.requires_grad = False
    return unet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "checkpoints", "ridl_policy_ppo_ds1.pth"),
                    help="thesis RIDL checkpoint holding the frozen UNet; falls back to HF SD-v1.5 weights if absent")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--threads", type=int, default=20)
    ap.add_argument("--tasks", nargs="*", default=ORDER, help="segments to extract, in priority order")
    ap.add_argument("--reverse", action="store_true", help="walk the job list backwards (second worker)")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    os.makedirs(CACHE, exist_ok=True)
    unet = load_unet(a.ckpt)
    tf = transforms.Compose([transforms.ToTensor(), transforms.Resize((64, 64))])
    norm = transforms.Normalize([0.5] * 3, [0.5] * 3)
    enc = torch.zeros(1, 1, unet.config.cross_attention_dim)

    jobs = [(s, c, g) for g in a.tasks for s in SUBJECTS for c in CONDS]
    if a.reverse:
        jobs = jobs[::-1]
    t_start = time.time()
    for i, (s, c, g) in enumerate(jobs):
        out = os.path.join(CACHE, seg_name(s, c, g) + ".npz")
        if os.path.exists(out):
            continue
        m = merged_segment(s, c, g)
        paths = m["image_path"].tolist()
        feats = np.zeros((len(paths), 16384), np.float16)
        for b in range(0, len(paths), a.batch):
            imgs = []
            for p in paths[b:b + a.batch]:
                rgb = cv2.cvtColor(cv2.imread(p, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                imgs.append(tf(rgb.astype(np.float32) / 255.0))
            x = norm(torch.stack(imgs))
            x = torch.cat([x, torch.zeros(x.size(0), 1, 64, 64)], 1)
            with torch.no_grad():
                o = unet(x, torch.zeros(x.size(0), dtype=torch.long),
                         enc.expand(x.size(0), -1, -1)).sample
            feats[b:b + len(imgs)] = o.reshape(x.size(0), -1).numpy().astype(np.float16)
        if os.path.exists(out):          # another worker finished it meanwhile
            continue
        tmpdir = os.path.join(os.path.dirname(CACHE), "tmp"); os.makedirs(tmpdir, exist_ok=True)
        tmp = os.path.join(tmpdir, f"part{os.getpid()}.npz")   # outside the cache dir, so never counted early
        np.savez(tmp, timestamp=m["timestamp"].to_numpy(), feats=feats, image_path=np.array(paths))
        os.replace(tmp, out)             # atomic: readers never see a partial file
        el = time.time() - t_start
        print(f"[{i+1}/{len(jobs)}] {seg_name(s, c, g)} n={len(paths)} elapsed={el/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
