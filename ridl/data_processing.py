import os
import numpy as np
import pandas as pd
from .dh_kinematics import solve_kinematics

def load_csv(path):
    return pd.read_csv(path)

def load_images(image_dir):
    """
    Scan the directory for left_*.png, extract timestamp,
    and return a DataFrame with columns ['timestamp','image_file','image_path'].
    """
    data = []
    for fn in os.listdir(image_dir):
        if fn.startswith("left_") and fn.endswith(".png"):
            ts = float(fn[5:-4])
            data.append((round(ts, 1), fn))
    df = pd.DataFrame(data, columns=['timestamp','image_file'])
    # full path so downstream code can open it
    df['image_path'] = df['image_file'].apply(lambda x: os.path.join(image_dir, x))
    return df.sort_values('timestamp').reset_index(drop=True)

def compute_flow_opt_rewards(df):
    y = df['y'].to_numpy()
    fx, fy, fz = [df[c].to_numpy() for c in ('fx','fy','fz')]
    roll, pitch, yaw = [df[c].to_numpy() for c in ('roll','pitch','yaw')]
    n = len(y)
    # Flow
    ymin, ymax = y.min(), y.max()
    #F = np.full(n,0.5) if np.isclose(ymin,ymax) else np.interp(y, (ymin,ymax), (0,1))
    F = df['F'].to_numpy()
    # Optimality (jerk-based)
    signals = [y, fx, fy, fz, roll, pitch, yaw]
    jerks   = np.vstack([np.abs(sig[1:]-sig[:-1]) for sig in signals])
    cj      = jerks.mean(axis=0)
    norm    = np.zeros_like(cj) if np.isclose(cj.min(),cj.max()) else (cj - cj.min())/(cj.max()-cj.min())
    O = np.concatenate(([1.0], 1 - norm))
    # Reward & colors
    R      = np.round(15*(1-np.abs(F-1)) + O/10, 2)
    colors = np.hstack([np.stack([1-F, F, O],1), np.ones((n,1))])
    return F, O, R, colors

def merge_and_process(csv_path, image_dir):
    # 1) Load
    df_data = load_csv(csv_path)
    df_img  = load_images(image_dir)

    # 2) Align timestamps
    df_data['timestamp'] = df_data['timestamp'].astype(float).round(1)
    df_img ['timestamp'] = df_img ['timestamp'].astype(float).round(1)

    # 3) Merge on timestamp, keep only rows with a matching image_path
    df = pd.merge(df_data, df_img, on='timestamp', how='left') \
           .dropna(subset=['image_path'])

    # 4) Drop rows missing any of the 10 signals
    df_clean = df.dropna(subset=[
        'x','y','z','roll','pitch','yaw','fx','fy','fz'
    ]).reset_index(drop=True)

    # 5) Compute F, O, R, colors
    F,O,R,colors = compute_flow_opt_rewards(df_clean)
    df_clean['Flow']       = F
    df_clean['Optimality'] = O
    df_clean['Reward']     = R

    # 6) Kinematics
    poses   = df_clean[['x','y','z','roll','pitch','yaw']].to_numpy()
    qs, jtraj = solve_kinematics(poses)

    # 7) Save results
    df_clean.to_csv('df_results.csv', index=False)
    np.save('jtraj.npy', jtraj)

    return df_clean, colors
