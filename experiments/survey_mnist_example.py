import io
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Any

# ==========================================
# 1. Setup & Dataclass Definition
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ClassLabel = Any

@dataclass
class Counterfactual:
    original_data: np.ndarray
    data: np.ndarray
    original_class: ClassLabel
    target_class: ClassLabel
    explainer: str
    metadata: dict = field(default_factory=dict)

    @staticmethod
    def _array_to_bytes(arr: np.ndarray) -> bytes:
        with io.BytesIO() as f:
            np.save(f, arr)
            return f.getvalue()

    @staticmethod
    def _bytes_to_array(b: bytes) -> np.ndarray:
        with io.BytesIO(b) as f:
            return np.load(f, allow_pickle=False)

    def serialize(self) -> bytes:
        return pickle.dumps({
            "original_data": Counterfactual._array_to_bytes(self.original_data),
            "data": Counterfactual._array_to_bytes(self.data),
            "original_class": self.original_class,
            "target_class": self.target_class,
            "explainer": self.explainer,
            "metadata": self.metadata,
        }, protocol=4)

    @staticmethod
    def deserialize(data: bytes) -> "Counterfactual":
        state = pickle.loads(data)
        return Counterfactual(
            original_data=Counterfactual._bytes_to_array(state["original_data"]),
            data=Counterfactual._bytes_to_array(state["data"]),
            original_class=state["original_class"],
            target_class=state["target_class"],
            explainer=state["explainer"],
            metadata=state.get("metadata", {})
        )

# ==========================================
# 2. PyTorch Model Definitions
# ==========================================
class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.2)
        )
    def forward(self, x):
        return self.block(x)

class TransConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.2)
        )
    def forward(self, x):
        return self.block(x)

class Encoder(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBlock(1, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, latent_dim),
            nn.Tanh()
        )
    def forward(self, x):
        return self.encoder(x)
    
class Decoder(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 256 * 4 * 4),
            nn.Tanh()
        )
        self.decoder = nn.Sequential(
            TransConvBlock(256, 128),
            TransConvBlock(128, 64),
            TransConvBlock(64, 32),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )
    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 256, 4, 4)
        x = self.decoder(x)
        return x
    
class Autoencoder(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def forward(self, x):
        return self.decoder(self.encoder(x))

# ==========================================
# 3. Plotting & Export Logic
# ==========================================
def process_and_export_cfe(cf, autoencoder, index, selected_lang):
    autoencoder.to(device)
    autoencoder.eval()

    # ---- RECONSTRUCTION ----
    org_latent = torch.tensor(cf.original_data, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        org_recon = autoencoder.decoder(org_latent)
    org_recon_np = org_recon.detach().cpu().numpy()[0, 0]
    org_recon_np = ((org_recon_np + 1) / 2).clip(0, 1)

    cf_latent = torch.tensor(cf.data, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        cfe_recon = autoencoder.decoder(cf_latent)
    cfe_recon_np = cfe_recon.detach().cpu().numpy()[0, 0]
    cfe_recon_np = ((cfe_recon_np + 1) / 2).clip(0, 1)

    # ---- OVERLAY DIFFERENCE MASK ----
    O = org_recon_np
    C = cfe_recon_np

    # Calculate stroke overlap, additions, and removals
    overlap = np.minimum(O, C)
    added = np.clip(C - O, 0, 1)    # Will be mapped to Green
    removed = np.clip(O - C, 0, 1)  # Will be mapped to Red

    # Start with pure white RGB
    diff_rgb = np.ones((*O.shape, 3))

    # 1. Overlap -> Black (Subtract from all channels)
    diff_rgb[:, :, 0] -= overlap
    diff_rgb[:, :, 1] -= overlap
    diff_rgb[:, :, 2] -= overlap

    # 2. Added -> Green (Subtract from Red and Blue channels)
    diff_rgb[:, :, 0] -= added
    diff_rgb[:, :, 2] -= added

    # 3. Removed -> Red (Subtract from Green and Blue channels)
    diff_rgb[:, :, 1] -= removed
    diff_rgb[:, :, 2] -= removed

    diff_rgb = np.clip(diff_rgb, 0, 1)

    # ---- EXPORT CONFIGURATIONS ----
    configs = [
        {"prefix": "en", "labels": ("Original", "Counterfactual", "Difference")},
        {"prefix": "pl", "labels": ("Oryginał", "Kontrfakt", "Różnica")}
    ]

    for cfg in configs:
        lbl_orig, lbl_cf, lbl_diff = cfg["labels"]
        
        fig = plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        plt.imshow(1.0 - org_recon_np, cmap='gray', vmin=0, vmax=1)
        plt.title(lbl_orig, fontsize=16)
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(1.0 - cfe_recon_np, cmap='gray', vmin=0, vmax=1)
        plt.title(lbl_cf, fontsize=16)
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(diff_rgb)
        plt.title(lbl_diff, fontsize=16)
        plt.axis("off")

        plt.tight_layout()
        
        # Always save both language versions to disk
        filename = f"{cfg['prefix']}_cfe_{index}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        
        # Only display the UI window for the language the user actually requested
        if cfg['prefix'] == selected_lang:
            print(f"Saved both versions. Displaying {selected_lang.upper()} view. Close window to exit.")
            plt.show()
        else:
            plt.close(fig) # Close the background figure so it doesn't stay in memory

# ==========================================
# 4. Execution Pipeline
# ==========================================
if __name__ == "__main__":
    lang_choice = ""
    while lang_choice not in ['en', 'pl']:
        lang_choice = input("Select view language / Wybierz język (en/pl): ").strip().lower()

    print("\nLoading Autoencoder...")
    autoencoder = Autoencoder(latent_dim=32).to(device)
    
    try:
        state_dict = torch.load('models/torch_ae_mnist_paper.pth', map_location=device)
        autoencoder.load_state_dict(state_dict)
    except Exception as e:
        print(f"Warning: Issue loading weights. Error: {e}")

    print("Loading Counterfactuals...")
    with open('results/mnist_tagged_cfes.pkl', 'rb') as f:
        cf_data_list = pickle.load(f)

    cfes = []
    for item in cf_data_list:
        if isinstance(item, dict) and 'cf_bytes' in item:
            cfes.append(Counterfactual.deserialize(item['cf_bytes']))
        elif isinstance(item, Counterfactual):
            cfes.append(item)

    target_idx = 11
    if target_idx < len(cfes):
        process_and_export_cfe(cfes[target_idx], autoencoder, target_idx, selected_lang=lang_choice)
    else:
        print(f"Error: Index {target_idx} is out of bounds. File only contains {len(cfes)} CFEs.")