#!/usr/bin/env python3
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import os
import time
import pandas as pd
import torch
import numpy as np
import cv2
from shutil import copyfile

# Force VisPy to use GLFW (no Qt)
import vispy.app
vispy.app.use_app('glfw')
from vispy import scene, app
from vispy.scene import visuals
from vispy.visuals.transforms import STTransform

from ridl.data_processing import load_csv, load_images, compute_flow_opt_rewards
from ridl.dataset import ConvImageDiffusionDataset
from ridl.model import RIDLPolicy

class OneDKalman:
    """1D Kalman filter for residual‐based error."""
    def __init__(self, q=1e-5, r=0.1):
        self.Q, self.R = q, r
        self.x, self.P = 0.0, 1.0

    def update(self, z):
        Pp = self.P + self.Q
        K  = Pp / (Pp + self.R)
        res = z - self.x
        self.x += K * res
        self.P  = (1 - K) * Pp
        return res

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def euler_to_rot(roll, pitch, yaw):
    Rx = np.array([[1,0,0],
                   [0,np.cos(roll),-np.sin(roll)],
                   [0,np.sin(roll), np.cos(roll)]], dtype=np.float32)
    Ry = np.array([[np.cos(pitch),0,np.sin(pitch)],
                   [0,1,0],
                   [-np.sin(pitch),0,np.cos(pitch)]], dtype=np.float32)
    Rz = np.array([[np.cos(yaw),-np.sin(yaw),0],
                   [np.sin(yaw), np.cos(yaw),0],
                   [0,0,1]], dtype=np.float32)
    return (Rz @ Ry @ Rx).astype(np.float32)

def load_policy(path, state_len, img_size, device, nc, act_dim):
    m = RIDLPolicy(
        nc=nc,
        nl=state_len,
        img_ch=3,
        img_size=tuple(img_size),
        act_dim=act_dim,
        hid_dim=512,
        n_encoder_layers=4,
        n_heads=8,
        ff_dim=256,
        dropout=0.1
    )
    sd = torch.load(path, map_location='cpu')
    m.load_state_dict(sd)
    return m.to(device).eval()

def test_ppo(csv_path, image_dir,
             ppo_model_path, ridl_model_path,
             img_size, output_dir):
    # 1) Load & merge CSV + images
    df_data = load_csv(csv_path)
    df_img  = load_images(image_dir)
    df_data['timestamp'] = df_data['timestamp'].astype(float).round(1)
    df_img ['timestamp'] = df_img ['timestamp'].astype(float).round(1)
    F, O, R_orig, _ = compute_flow_opt_rewards(df_data)
    df_data['Flow'], df_data['Optimality'], df_data['Reward_orig'] = F, O, R_orig

    df = (pd.merge(df_data, df_img, on='timestamp', how='left')
          .dropna(subset=['image_path','x','y','z','roll','pitch','yaw','fx','fy','fz'])
          .reset_index(drop=True)
          .rename(columns={'image_path':'image'}))

    # limit to first 7 minutes
    t0 = df['timestamp'].iloc[0]
    df = df[df['timestamp'] <= t0 + 420.0].reset_index(drop=True)
    N = len(df) - 1
    print(f"Using {N+1} frames → {N} steps (≤{t0+420:.1f}s)")

    # 2) Prepare dataset
    state_feats = ['Flow','Optimality']
    traj_feats  = ['x','y','z','roll','pitch','yaw','fx','fy','fz']
    dataset = ConvImageDiffusionDataset(df,
        state_feats=state_feats,
        action_feats=traj_feats,
        img_size=tuple(img_size)
    )

    # 3) Load models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    ppo_model  = load_policy(ppo_model_path,
                             dataset.state_len, img_size, device,
                             nc=len(state_feats), act_dim=len(traj_feats))
    ridl_model = load_policy(ridl_model_path,
                             dataset.state_len, img_size, device,
                             nc=len(state_feats), act_dim=len(traj_feats))

    # 4) Build VisPy canvas + grid
    canvas = scene.SceneCanvas(keys='interactive',
                               title='PPO vs RIDL Comparison')
    canvas.show()
    grid = canvas.central_widget.add_grid(spacing=10)

    view3d   = grid.add_view(row=0, col=0, row_span=2)
    view3d.camera = 'turntable'
    visuals.XYZAxis(parent=view3d.scene)

    view_err = grid.add_view(row=0, col=1)
    view_err.camera = scene.PanZoomCamera(aspect=1)

    view_img = grid.add_view(row=1, col=1)
    view_img.camera = scene.PanZoomCamera(aspect=1)
    h, w = img_size
    view_img.camera.set_range(x=(0, w), y=(0, h))

    # 5) Create visuals
    def mk_line3d(col):
        return visuals.Line(pos=np.zeros((1,3),dtype=np.float32),
                            color=col, parent=view3d.scene,
                            method='gl', width=2)
    true_line = mk_line3d('white')
    ppo_line  = mk_line3d('red')
    ridl_line = mk_line3d('blue')

    roll_vis  = visuals.Line(pos=np.zeros((2,3),dtype=np.float32),
                             color='red', parent=view3d.scene, method='gl', width=3)
    pitch_vis = visuals.Line(pos=np.zeros((2,3),dtype=np.float32),
                             color='green', parent=view3d.scene, method='gl', width=3)
    yaw_vis   = visuals.Line(pos=np.zeros((2,3),dtype=np.float32),
                             color='yellow', parent=view3d.scene, method='gl', width=3)

    img_vis = visuals.Image(np.zeros((h, w, 3),dtype=np.float32),
                            origin='upper-left',
                            parent=view_img.scene,
                            method='auto')
    img_vis.transform = STTransform()

    err_true_vis = visuals.Line(pos=np.zeros((1,2),dtype=np.float32),
                                color='white', parent=view_err.scene, method='gl', width=2)
    err_pred_vis = visuals.Line(pos=np.zeros((1,2),dtype=np.float32),
                                color='blue',  parent=view_err.scene, method='gl', width=2)

    # 6) Spatial scaling
    xs, ys, zs = df['x'], df['y'], df['z']
    span   = np.array([xs.max()-xs.min(), ys.max()-ys.min(), zs.max()-zs.min()], dtype=np.float32)
    center = np.array([xs.min()+span[0]/2, ys.min()+span[1]/2, zs.min()+span[2]/2], dtype=np.float32)
    view3d.camera.center   = center.tolist()
    view3d.camera.distance = float(np.linalg.norm(span)*1.5)
    orient_scale = 1.0

    # 7) Kalman filters + histories
    kf_t = [OneDKalman() for _ in range(3)]
    kf_r = [OneDKalman() for _ in range(3)]
    hist_t,   hist_pp, hist_rr = [], [], []
    hist_et,  hist_er         = [], []
    vis_t,    vis_pp,  vis_rr  = [], [], []
    vis_et,   vis_er          = [], []
    lastvis = df['timestamp'].iloc[0]

    print("Starting inference…")
    for i in range(N):
        # get raw image
        raw = dataset.raw_images[i]
        raw = cv2.resize(raw, (h, w), interpolation=cv2.INTER_AREA)

        # ensure it's 8-bit BGR for Canny
        if raw.dtype != np.uint8:
            # assume float in [0,1]
            raw_uint8 = (raw * 255).clip(0,255).astype(np.uint8)
        else:
            raw_uint8 = raw

        # convert to grayscale & apply Canny
        #gray = cv2.cvtColor(raw_uint8, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(raw_uint8, 100, 200)


        # stack to 3-channels for VisPy and normalize
        edges_vis = np.stack([edges, edges, edges], axis=-1).astype(np.float32) / 255.0
        img_vis.set_data(edges_vis)

        # predict actions
        state = dataset.states[i].unsqueeze(0).to(device)
        inp   = dataset.transform(raw).unsqueeze(0).to(device)
        with torch.no_grad():
            mu_p,_,_ = ppo_model(state, inp)
            mu_r,_,_ = ridl_model(state, inp)
        dp = mu_p.cpu().numpy().flatten().astype(np.float32)
        dr = mu_r.cpu().numpy().flatten().astype(np.float32)

        # true vs predicted positions
        cur_p  = df.loc[i,   ['x','y','z']].to_numpy(dtype=np.float32)
        true_n = df.loc[i+1, ['x','y','z']].to_numpy(dtype=np.float32)
        pred_p = cur_p + dp[:3]
        ori_r  = df.loc[i, ['roll','pitch','yaw']].to_numpy(dtype=np.float32) + dr[3:6]

        # update histories & Kalman
        hist_t.append(true_n); hist_pp.append(pred_p); hist_rr.append(pred_p)
        res_t = np.array([kf_t[j].update(true_n[j]) for j in range(3)], dtype=np.float32)
        res_r = np.array([kf_r[j].update(pred_p[j]) for j in range(3)], dtype=np.float32)
        et, er = np.linalg.norm(res_t), np.linalg.norm(res_r)
        hist_et.append(et); hist_er.append(er)

        ts = df['timestamp'].iloc[i]
        if i == 0 or ts - lastvis >= 1.0:
            lastvis = ts
            vis_t.append(true_n); vis_pp.append(pred_p); vis_rr.append(pred_p)
            vis_et.append(et);      vis_er.append(er)

            true_line .set_data(np.array(vis_t),  connect='strip')
            ppo_line  .set_data(np.array(vis_pp), connect='strip')
            ridl_line .set_data(np.array(vis_rr), connect='strip')

            x, y, z = pred_p
            R = euler_to_rot(*ori_r)
            roll_vis  .set_data(np.array([[x,y,z],
                                          [x+R[0,0]*orient_scale,
                                           y+R[1,0]*orient_scale,
                                           z+R[2,0]*orient_scale]]))
            pitch_vis .set_data(np.array([[x,y,z],
                                          [x+R[0,1]*orient_scale,
                                           y+R[1,1]*orient_scale,
                                           z+R[2,1]*orient_scale]]))
            yaw_vis   .set_data(np.array([[x,y,z],
                                          [x+R[0,2]*orient_scale,
                                           y+R[1,2]*orient_scale,
                                           z+R[2,2]*orient_scale]]))

            steps = np.arange(len(vis_et), dtype=np.float32)
            err_true_vis.set_data(np.column_stack((steps,vis_et)), connect='strip')
            err_pred_vis.set_data(np.column_stack((steps,vis_er)), connect='strip')
            view_err.camera.set_range(x=(0,len(vis_et)), y=(min(vis_et+vis_er), max(vis_et+vis_er)))

        # refresh
        canvas.update()
        app.process_events()
        time.sleep(0.005)

    # 8) Save CSV + images
    df_out = pd.DataFrame({
        'true_x':   [t[0] for t in hist_t],
        'true_y':   [t[1] for t in hist_t],
        'true_z':   [t[2] for t in hist_t],
        'ppo_x':    [p[0] for p in hist_pp],
        'ppo_y':    [p[1] for p in hist_pp],
        'ppo_z':    [p[2] for p in hist_pp],
        'ridl_x':   [r[0] for r in hist_rr],
        'ridl_y':   [r[1] for r in hist_rr],
        'ridl_z':   [r[2] for r in hist_rr],
        'err_true': hist_et,
        'err_ridl': hist_er
    })
    out_csv = os.path.join(output_dir, 'comparison.csv')
    df_out.to_csv(out_csv, index=False)
    print("Saved CSV →", out_csv)

    imgs_out = os.path.join(output_dir, 'images')
    ensure_dir(imgs_out)
    for src in df['image'][:len(hist_t)]:
        copyfile(src, os.path.join(imgs_out, os.path.basename(src)))

def parse_args():
    p = argparse.ArgumentParser(description="PPO vs RIDL Real-time Comparison")
    p.add_argument('--csv',        required=True, help='Data CSV')
    p.add_argument('--image-dir',  required=True, help='Image folder')
    p.add_argument('--ppo-model',  required=True, help='PPO .pth')
    p.add_argument('--ridl-model', required=True, help='RIDL .pth')
    p.add_argument('--output-dir', default='outputs', help='Output dir')
    p.add_argument('--img-size',   nargs=2, type=int, default=[64,64], help='H W')
    return p.parse_args()

if __name__ == '__main__':
    args = parse_args()
    ensure_dir(args.output_dir)
    test_ppo(args.csv, args.image_dir,
             args.ppo_model, args.ridl_model,
             args.img_size, args.output_dir)
    print("Entering interactive GUI loop…")
    app.run()
