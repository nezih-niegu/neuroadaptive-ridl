import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import UNet2DConditionModel
from torchvision.transforms import Resize, Normalize
import math

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer input"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        return x + self.pe[:, :x.size(1), :]
        
class RIDLPolicy(nn.Module):
    def __init__(self, 
                 nc,                         # Number of numeric channels
                 nl,                         # Length of numeric sequence
                 img_ch=3,                   # Image channels
                 img_size=(32, 32),          # Image size
                 act_dim=9,                  # Action dimension
                 hid_dim=512,                # Hidden dimension
                 n_encoder_layers=4,         # Number of transformer encoder layers
                 n_heads=8,                  # Number of attention heads
                 ff_dim=3200,                # Feedforward dimension
                 dropout=0.1,                # Dropout rate
                 device="cpu"):
        super().__init__()
        
        # --- Stable Diffusion UNet only (no VAE) ---
        self.unet = UNet2DConditionModel.from_pretrained(
            "runwayml/stable-diffusion-v1-5", 
            subfolder="unet"
        ).to(device).eval()
        
        # Freeze UNet weights
        for p in self.unet.parameters():
            p.requires_grad = False
            
        # Preprocessing transforms
        self.resize = Resize(img_size)
        self.normalize = Normalize([0.5]*img_ch, [0.5]*img_ch)
        
        # Embedding layer for numeric inputs
        self.num_embed = nn.Conv1d(nc, hid_dim, kernel_size=1)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(hid_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hid_dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu"
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=n_encoder_layers
        )
        
        # Calculate UNet output dimension
        with torch.no_grad():
            dummy_img = torch.zeros(1, img_ch, *img_size, device=device)
            x = self.normalize(self.resize(dummy_img))
            # pad to 4 channels
            pad = torch.zeros(1, 1, *img_size, device=device)
            unet_in = torch.cat([x, pad], dim=1)
            timestep = torch.zeros(1, dtype=torch.long, device=device)
            encoder_states = torch.zeros(1, 1, self.unet.config.cross_attention_dim, device=device)
            unet_out = self.unet(unet_in, timestep, encoder_states).sample
            flat_img = unet_out.view(1, -1).size(1)
        
        # Project UNet features to match transformer dimension
        self.unet_projection = nn.Linear(flat_img, hid_dim)
        
        # Final projection layer
        self.projection = nn.Sequential(
            nn.Linear(hid_dim, hid_dim//2),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Policy and value heads
        self.mean_head = nn.Linear(hid_dim//2, act_dim)
        self.value_head = nn.Linear(hid_dim//2, 1)
        self.log_std = nn.Parameter(torch.zeros(act_dim))
        
    def forward(self, numeric, img):
        device = next(self.parameters()).device
        batch_size = numeric.size(0)
        
        # Process numeric input with transformer
        # [batch, channels, length] -> [batch, length, hidden_dim]
        x_num = self.num_embed(numeric.to(device)).transpose(1, 2)
        
        # Add positional encoding
        x_num = self.pos_encoder(x_num)
        
        # Apply transformer encoder
        # Create a mask for padding if needed (assuming all sequence is valid)
        src_mask = None
        x_num = self.transformer_encoder(x_num, src_mask)
        
        # Pool transformer outputs (mean pooling over sequence length)
        x_num = x_num.mean(dim=1)  # [batch, hidden_dim]
        
        # Process image with UNet
        x_img = self.normalize(self.resize(img)).to(device)
        pad = torch.zeros(x_img.size(0), 1, *x_img.shape[2:], device=device)
        unet_in = torch.cat([x_img, pad], dim=1)
        
        with torch.no_grad():
            timestep = torch.zeros(x_img.size(0), dtype=torch.long, device=device)
            encoder_states = torch.zeros(x_img.size(0), 1, 
                                       self.unet.config.cross_attention_dim, 
                                       device=device)
            unet_feat = self.unet(unet_in, timestep, encoder_states).sample
            img_feat = unet_feat.view(batch_size, -1)
        
        # Project image features to match transformer dimension
        img_feat = self.unet_projection(img_feat)  # [batch, hidden_dim]
        
        # Combine image and numeric features
        # Here we use element-wise addition, could also use concatenation
        combined = x_num + img_feat
        
        # Final projection
        x = self.projection(combined)
        
        # Output heads
        mu = self.mean_head(x)
        sigma = torch.exp(self.log_std).unsqueeze(0).expand_as(mu)
        value = self.value_head(x).squeeze(-1)
        
        return mu, sigma, value