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
def draw_arrow_with_text(ax, text):
    """Helper to draw a styled arrow with centered text above it."""
    ax.axis('off')
    ax.annotate('', xy=(0.95, 0.5), xytext=(0.05, 0.5),
                arrowprops=dict(facecolor='black', width=3, headwidth=12),
                xycoords='axes fraction')
    ax.text(0.5, 0.65, text, ha='center', va='center', fontsize=12, fontweight='bold', transform=ax.transAxes)

def draw_class_label(ax, label):
    """Helper to draw the large prediction number."""
    ax.axis('off')
    ax.text(0.5, 0.5, str(label), ha='center', va='center', fontsize=54, fontweight='bold', transform=ax.transAxes)

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

    org_class = cf.original_class
    target_class = cf.target_class

    # ---- EXPORT CONFIGURATIONS ----
    configs = [
        {
            "prefix": "en",
            "pred_text": "Model\nPrediction",
            "expl_text": "Explainer\n(What if?)"
        },
        {
            "prefix": "pl",
            "pred_text": "Decyzja\nkomputera",
            "expl_text": "Wyjaśnienie\n(Co by było?)"
        }
    ]

    for cfg in configs:
        # ---------------------------------------------------------
        # FIGURE 1: Standard Prediction
        # ---------------------------------------------------------
        fig1 = plt.figure(figsize=(7, 2.5))
        
        ax1 = fig1.add_subplot(1, 3, 1)
        ax1.imshow(1.0 - org_recon_np, cmap='gray', vmin=0, vmax=1)
        ax1.axis("off")
        
        ax2 = fig1.add_subplot(1, 3, 2)
        draw_arrow_with_text(ax2, cfg["pred_text"])
        
        ax3 = fig1.add_subplot(1, 3, 3)
        draw_class_label(ax3, org_class)
        
        plt.tight_layout()
        fig1.savefig(f"{cfg['prefix']}_fig1_standard_pred_{index}.png", dpi=300, bbox_inches='tight')

        # ---------------------------------------------------------
        # FIGURE 2: Counterfactual Explanation
        # ---------------------------------------------------------
        fig2 = plt.figure(figsize=(12, 2.5))
        
        bx1 = fig2.add_subplot(1, 5, 1)
        bx1.imshow(1.0 - org_recon_np, cmap='gray', vmin=0, vmax=1)
        bx1.axis("off")
        
        bx2 = fig2.add_subplot(1, 5, 2)
        draw_arrow_with_text(bx2, cfg["expl_text"])
        
        bx3 = fig2.add_subplot(1, 5, 3)
        bx3.imshow(1.0 - cfe_recon_np, cmap='gray', vmin=0, vmax=1)
        bx3.axis("off")
        
        bx4 = fig2.add_subplot(1, 5, 4)
        draw_arrow_with_text(bx4, cfg["pred_text"])
        
        bx5 = fig2.add_subplot(1, 5, 5)
        draw_class_label(bx5, target_class)
        
        plt.tight_layout()
        fig2.savefig(f"{cfg['prefix']}_fig2_counterfactual_{index}.png", dpi=300, bbox_inches='tight')

        # Only keep the figure open in memory if it matches the selected language
        if cfg['prefix'] != selected_lang:
            plt.close(fig1)
            plt.close(fig2)

    print(f"\nSaved all image versions to disk.")
    print(f"Displaying {selected_lang.upper()} view. Close both figure windows to exit.")
    plt.show()

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
    try:
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
            
    except FileNotFoundError:
        print("Error: Could not find 'results/mnist_tagged_cfes.pkl'. Please check the path.")