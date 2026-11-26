#!/usr/bin/env python3
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse, os
import time
import pandas as pd
import torch
import numpy as np
import cv2
from shutil import copyfile

# Force VisPy to use GLFW (no Qt)
import vispy.app
vispy.app.use_app('glfw')
from vispy import scene, app, visuals

from ridl.data_processing import load_csv, load_images, compute_flow_opt_rewards
from ridl.dataset import ConvImageDiffusionDataset
from ridl.model import RIDLPolicy

class OneDKalman:
    """1D Kalman filter for residual-based error."""
    def __init__(self, q=1e-5, r=0.1):
        self.Q, self.R = q, r
        self.x, self.P = 0.0, 1.0
    def update(self, z):
        Pp = self.P + self.Q
        K  = Pp / (Pp + self.R)
        res = z - self.x
        self.x += K*res
        self.P  = (1 - K)*Pp
        return res

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

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def test_ppo(csv_path, image_dir, ppo_model_path, ridl_model_path,
             img_size, output_dir, device_str):
    # ─── Load & merge ────────────────────────────────────────────────
    df_data = load_csv(csv_path)
    df_img  = load_images(image_dir)
    df_data['timestamp'] = df_data['timestamp'].astype(float).round(1)
    df_img ['timestamp'] = df_img ['timestamp'].astype(float).round(1)
    F,O,R_orig,_ = compute_flow_opt_rewards(df_data)
    df_data['Flow'],df_data['Optimality'],df_data['Reward_orig'] = F,O,R_orig

    df = (pd.merge(df_data, df_img, on='timestamp', how='left')
          .dropna(subset=['image_path','x','y','z','roll','pitch','yaw','fx','fy','fz'])
          .reset_index(drop=True)
          .rename(columns={'image_path':'image'}))

    # ─── Trim to first 7 min ───────────────────────────────────────────
    t0 = df['timestamp'].iloc[0]
    df = df[df['timestamp'] <= t0 + 420.0].reset_index(drop=True)
    N  = len(df) - 1
    print(f"Using {N+1} frames → {N} steps (≤{t0+420:.1f}s)")

    # ─── Dataset & models ─────────────────────────────────────────────
    state_feats = ['Flow','Optimality']
    traj_feats  = ['x','y','z','roll','pitch','yaw','fx','fy','fz']
    dataset = ConvImageDiffusionDataset(df,
        state_feats=state_feats,
        action_feats=traj_feats,
        img_size=tuple(img_size)
    )
    device = torch.device(device_str)
    print("Using device:", device)

    def load_policy(path):
        m = RIDLPolicy(
            nc=len(state_feats), nl=dataset.state_len,
            img_ch=3, img_size=tuple(img_size),
            act_dim=len(traj_feats),
            hid_dim=512, n_encoder_layers=4,
            n_heads=8, ff_dim=256, dropout=0.1
        )
        sd = torch.load(path, map_location='cpu')
        m.load_state_dict(sd)
        return m.to(device).eval()

    ppo_model  = load_policy(ppo_model_path)
    ridl_model = load_policy(ridl_model_path)

    # ─── VisPy Window A: 3D Trajectory ─────────────────────────────────
    canvas_traj = scene.SceneCanvas(keys='interactive',
                                    title=f'Trajectory @ t={t0:.1f}s',
                                    show=True)
    view_traj = canvas_traj.central_widget.add_view()
    view_traj.camera = 'turntable'
    visuals.XYZAxis(parent=view_traj.scene)
    true_line = visuals.Line(color='white', parent=view_traj.scene, connect='strip')
    ppo_line  = visuals.Line(color='red',   parent=view_traj.scene, connect='strip')
    ridl_line = visuals.Line(color='blue',  parent=view_traj.scene, connect='strip')
    roll_vis  = visuals.Line(color='red',   parent=view_traj.scene)
    pitch_vis = visuals.Line(color='green', parent=view_traj.scene)
    yaw_vis   = visuals.Line(color='yellow',parent=view_traj.scene)
    xs, ys, zs = df['x'], df['y'], df['z']
    scale = np.linalg.norm([xs.max()-xs.min(),
                            ys.max()-ys.min(),
                            zs.max()-zs.min()]) * 0.1

    # ─── VisPy Window B: Error Curves ──────────────────────────────────
    canvas_err = scene.SceneCanvas(keys='interactive',
                                   title=f'Errors @ t={t0:.1f}s',
                                   show=True)
    view_err = canvas_err.central_widget.add_view()
    view_err.camera = scene.PanZoomCamera(aspect=1)
    view_err.camera.set_range(x=(0,N), y=(0,1))
    err_true_vis = visuals.Line(color='white', parent=view_err.scene)
    err_ppo_vis  = visuals.Line(color='red',   parent=view_err.scene)
    err_ridl_vis = visuals.Line(color='blue',  parent=view_err.scene)

    # ─── OpenCV Filter Windows ────────────────────────────────────────
    filter_funcs = {
        'Raw RGB':   lambda bgr: cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB),
        'Gray':      lambda bgr: cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY),
        'Gaussian':  lambda bgr: cv2.GaussianBlur(cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY),(5,5),0),
        'Canny':     lambda bgr: cv2.Canny(cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY),100,200),
        'Sobel':     lambda bgr: np.hypot(
                         *np.gradient(cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY).astype(np.float32))),
        'Laplacian': lambda bgr: cv2.Laplacian(
                         cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY),cv2.CV_32F)
    }
    for name in filter_funcs:
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)

    # ─── Kalman & buffers ─────────────────────────────────────────────
    kf_t  = [OneDKalman() for _ in range(3)]
    kf_pp = [OneDKalman() for _ in range(3)]
    kf_r  = [OneDKalman() for _ in range(3)]
    trues, pp, rr = [], [], []
    et, epp, er   = [], [], []
    first         = True

    print("Starting inference…")
    for i in range(N):
        t = df.loc[i,'timestamp']
        print(f"[Frame {i+1}/{N}] t={t:.1f}s")

        # ─── Load frame & show filters ───────────────────────────────
        img_path = df.loc[i,'image']
        bgr      = cv2.imread(img_path, cv2.IMREAD_COLOR)
        for name, func in filter_funcs.items():
            out = func(bgr)
            # convert float images back to uint8 for imshow if needed
            if out.dtype != np.uint8:
                out = cv2.normalize(out, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            cv2.imshow(name, out)

        # ─── PPO vs RIDL inference ───────────────────────────────────
        rgb   = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        state = dataset.states[i].unsqueeze(0).to(device)
        inp   = dataset.transform(rgb).unsqueeze(0).to(device)
        with torch.no_grad():
            mu_p,_,_ = ppo_model(state, inp)
            mu_r,_,_ = ridl_model(state, inp)

        a_p, a_r = mu_p.cpu().numpy().flatten(), mu_r.cpu().numpy().flatten()
        true_n   = df.loc[i+1, ['x','y','z']].to_numpy()
        trues.append(true_n); pp.append(a_p[:3]); rr.append(a_r[:3])

        # ─── Compute residuals & update VisPy ───────────────────────
        res_t = np.array([kf_t[j].update(true_n[j]) for j in range(3)])
        res_p = np.array([kf_pp[j].update(a_p[j])   for j in range(3)])
        res_r = np.array([kf_r[j].update(a_r[j])    for j in range(3)])
        et.append(np.linalg.norm(res_t))
        epp.append(np.linalg.norm(res_p))
        er.append(np.linalg.norm(res_r))

        # Trajectory
        true_line.set_data(np.array(trues))
        ppo_line .set_data(np.array(pp))
        ridl_line.set_data(np.array(rr))
        x,y,z,roll,pitch,yaw = *a_r[:3], *a_r[3:6]
        R = euler_to_rot(roll,pitch,yaw)
        roll_vis .set_data([[x,y,z],[x+R[0,0]*scale,y+R[1,0]*scale,z+R[2,0]*scale]])
        pitch_vis.set_data([[x,y,z],[x+R[0,1]*scale,y+R[1,1]*scale,z+R[2,1]*scale]])
        yaw_vis  .set_data([[x,y,z],[x+R[0,2]*scale,y+R[1,2]*scale,z+R[2,2]*scale]])
        canvas_traj.title = f'Trajectory @ t={t:.1f}s'
        canvas_traj.update()

        # Errors
        steps = np.arange(len(et))
        err_true_vis.set_data(np.column_stack((steps, et)))
        err_ppo_vis .set_data(np.column_stack((steps, epp)))
        err_ridl_vis.set_data(np.column_stack((steps, er)))
        canvas_err.title = f'Errors @ t={t:.1f}s'
        canvas_err.update()

        # ─── Process events for both GLFW & OpenCV ────────────────
        app.process_events()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # ─── Cleanup & save ───────────────────────────────────────────
    cv2.destroyAllWindows()
    df_out = pd.DataFrame({
        'true_x':  [v[0] for v in trues],
        'true_y':  [v[1] for v in trues],
        'true_z':  [v[2] for v in trues],
        'ppo_x':   [v[0] for v in pp],
        'ppo_y':   [v[1] for v in pp],
        'ppo_z':   [v[2] for v in pp],
        'ridl_x':  [v[0] for v in rr],
        'ridl_y':  [v[1] for v in rr],
        'ridl_z':  [v[2] for v in rr],
        'err_true':et,
        'err_ppo':  epp,
        'err_ridl':er
    })
    out_csv = os.path.join(output_dir, 'comparison.csv')
    df_out.to_csv(out_csv, index=False)
    print("Saved CSV:", out_csv)

    imgs_out = os.path.join(output_dir, 'images')
    ensure_dir(imgs_out)
    for img in df['image'][:len(trues)]:
        copyfile(img, os.path.join(imgs_out, os.path.basename(img)))

    return out_csv

def parse_args():
    p = argparse.ArgumentParser(description="Real-time PPO vs RIDL Comparison")
    p.add_argument('--csv',        required=True,  help='data CSV')
    p.add_argument('--image-dir',  required=True,  help='images folder')
    p.add_argument('--ppo-model',  required=True,  help='PPO .pth')
    p.add_argument('--ridl-model', required=True,  help='RIDL .pth')
    p.add_argument('--output-dir', default='outputs', help='output folder')
    p.add_argument('--img-size',   nargs=2, type=int, default=[64,64], help='H W')
    p.add_argument('--device',     choices=['cpu','cuda'], default='cpu', help='device')
    return p.parse_args()

def main():
    args = parse_args()
    ensure_dir(args.output_dir)
    test_ppo(args.csv, args.image_dir,
             args.ppo_model, args.ridl_model,
             args.img_size, args.output_dir,
             args.device)
    print("Done.")

if __name__ == '__main__':
    main()
