import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms

class ConvImageDiffusionDataset(Dataset):
    def __init__(
        self, df, state_feats, action_feats,
        image_col='image',
        img_diff_threshold=1e-2,
        state_channels=2, state_dim_per_channel=8,
        img_size=(64,64)
    ):
        # Store state dimensions for external usage
        self.state_channels = state_channels
        self.state_len = state_dim_per_channel

        # Normalize numeric features
        S = df[state_feats].values.astype(np.float32)
        A = df[action_feats].values.astype(np.float32)
        self.s_mean, self.s_std = S.mean(0), S.std(0) + 1e-6
        self.a_mean, self.a_std = A.mean(0), A.std(0) + 1e-6
        S = (S - self.s_mean) / self.s_std
        A = (A - self.a_mean) / self.a_std

        # Reshape numeric into [C, L]
        N, F_dim = S.shape
        needed = state_channels * state_dim_per_channel
        S_tensor = torch.FloatTensor(S)
        # pad if needed
        if F_dim < needed:
            pad = torch.zeros(N, needed - F_dim, dtype=torch.float32)
            S_tensor = torch.cat([S_tensor, pad], dim=1)
        # reshape into channels sequence
        S_tensor = S_tensor.reshape(N, state_channels, -1)
        # interpolate to exact length if necessary
        if S_tensor.size(2) != state_dim_per_channel:
            S_tensor = F.interpolate(
                S_tensor,
                size=state_dim_per_channel,
                mode='linear',
                align_corners=False
            )
        self.states = S_tensor
        self.actions = torch.FloatTensor(A)

        # Prepare raw images (load from disk) and change detection
        raw_paths = df[image_col].tolist()
        self.raw_images, prev = [], None
        for path in raw_paths:
            # load image from file
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(f"Image not found: {path}")
            # convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            arr = img.astype(np.float32) / 255.0

            # change-detect
            if prev is None or np.mean(np.abs(arr - prev)) > img_diff_threshold:
                prev = arr
            self.raw_images.append(prev)

        # On-the-fly transform pipeline
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(img_size),
        ])

    def __len__(self):
        return self.actions.size(0)

    def __getitem__(self, idx):
        numeric = self.states[idx]
        action  = self.actions[idx]
        image_np= self.raw_images[idx]
        image   = self.transform(image_np)
        return {
            'numeric': numeric,
            'image':   image,
            'action':  action,
        }
