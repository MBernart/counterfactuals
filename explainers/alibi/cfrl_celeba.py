import numpy as np
import torch
import torch.nn as nn
import os
from alibi.explainers import CounterfactualRL
from explainers_lib.explainers.celery_remote import app, create_celery_tasks
from explainers_lib import Explainer, Dataset, Model, Counterfactual

# --- Autoencoder and CNN Architectures from experiments/celebA_ensemble.ipynb ---


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.block(x)


class TransConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBlock(3, 32),  # 128 -> 64
            ConvBlock(32, 64),  # 64 -> 32
            ConvBlock(64, 128),  # 32 -> 16
            ConvBlock(128, 256),  # 16 -> 8
            ConvBlock(256, 512),  # 8 -> 4
            ConvBlock(512, 512),  # 4 -> 2
            nn.Flatten(),
            nn.Linear(512 * 2 * 2, latent_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()

        self.fc = nn.Sequential(nn.Linear(latent_dim, 512 * 2 * 2), nn.Tanh())

        self.decoder = nn.Sequential(
            TransConvBlock(512, 512),  # 2 → 4
            TransConvBlock(512, 512),  # 4 → 8
            TransConvBlock(512, 256),  # 8 → 16
            TransConvBlock(256, 128),  # 16 → 32
            TransConvBlock(128, 64),  # 32 → 64
            TransConvBlock(64, 32),  # 64 → 128
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 512, 2, 2)
        x = self.decoder(x)
        return x


class Autoencoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def forward(self, x):
        return self.decoder(self.encoder(x))


# --- CFRL Wrapper for CelebA ---


class CFRLCelebA(Explainer):
    def __init__(
        self,
        latent_dim=128,
        coeff_sparsity=0.5,
        coeff_consistency=0.5,
        train_steps=25_000,
        batch_size=32,
    ):
        self.latent_dim = latent_dim
        self.coeff_sparsity = coeff_sparsity
        self.coeff_consistency = coeff_consistency
        self.train_steps = train_steps
        self.batch_size = batch_size
        self.explainer = None
        # We assume a CUDA device if available, but Alibi might handle device internally or require CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.autoencoder = None  # Store autoencoder for use in explain

    def fit(self, model: Model, data: Dataset) -> None:
        self.model = model
        # Load pre-trained Autoencoder
        possible_paths = [
            "experiments/autoencoder_celeba_best.pth",
            "../../experiments/autoencoder_celeba_best.pth",
            "/home/patryk/Desktop/counterfactuals/experiments/autoencoder_celeba_best.pth",
        ]

        ae_path = None
        for p in possible_paths:
            if os.path.exists(p):
                ae_path = p
                break

        if ae_path is None:
            raise FileNotFoundError(
                "Could not find 'autoencoder_celeba_best.pth' in expected locations."
            )

        self.autoencoder = Autoencoder(latent_dim=self.latent_dim).to(self.device)
        weights = torch.load(ae_path, map_location=self.device)
        self.autoencoder.load_state_dict(weights)
        self.autoencoder.eval()

        # Check input data shape
        is_latent_input = False
        if hasattr(data.data, "ndim") and data.data.ndim == 2:
            is_latent_input = True
        elif hasattr(data.data, "shape") and len(data.data.shape) == 2:
            is_latent_input = True

        X_train = data.data

        if is_latent_input:
            print(
                "CFRLCelebA: Detected 2D input (latent). Reconstructing images for training..."
            )
            # Reconstruct images from latent vectors
            # Warning: This expands dataset to images in memory.
            with torch.no_grad():
                latent_tensor = torch.tensor(data.data, dtype=torch.float32).to(
                    self.device
                )
                # Process in batches to avoid OOM during reconstruction if dataset is huge
                # though we will store result in RAM anyway.
                recon_images = []
                bs = 100
                for i in range(0, len(latent_tensor), bs):
                    batch = latent_tensor[i : i + bs]
                    recon = self.autoencoder.decoder(batch)
                    recon_images.append(recon.cpu().numpy())

                X_train = np.concatenate(recon_images, axis=0)
                print(
                    f"CFRLCelebA: Reconstructed {len(X_train)} images with shape {X_train.shape[1:]}."
                )

            # Predictor Wrapper for Latent Input Model:
            # Image -> Encoder -> Latent -> User Model
            predictor_wrapper = self._latent_predictor_wrapper
        else:
            predictor_wrapper = self._regular_predictor_wrapper

        # Initialize CounterfactualRL with Real AE
        self.explainer = CounterfactualRL(
            predictor=predictor_wrapper,
            encoder=self.autoencoder.encoder,
            decoder=self.autoencoder.decoder,
            latent_dim=self.latent_dim,
            coeff_sparsity=self.coeff_sparsity,
            coeff_consistency=self.coeff_consistency,
            train_steps=self.train_steps,
            batch_size=self.batch_size,
            backend="pytorch",
        )

        if os.path.exists("celeba_cfrl.dill"):
            print(
                "CFRLCelebA: Found existing explainer file 'celeba_cfrl.dill'. Loading it instead of fitting anew."
            )
            self.explainer.load("celeba_cfrl.dill", predictor=predictor_wrapper)
        else:
            self.explainer.fit(X=X_train)

        self.explainer.save("celeba_cfrl.dill")

    def _regular_predictor_wrapper(self, x: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(x)

    def _latent_predictor_wrapper(self, x: np.ndarray) -> np.ndarray:
        # x is (N, 3, 128, 128)
        with torch.no_grad():
            x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)
            z = self.autoencoder.encoder(x_tensor).cpu().numpy()
        # Pass latent z to user model
        return self.model.predict_proba(z)

    def explain(self, model: Model, data: Dataset) -> list[Counterfactual]:
        cfs = list()

        # Check input data shape
        is_latent_input = False
        if hasattr(data.data, "ndim") and data.data.ndim == 2:
            is_latent_input = True
        elif hasattr(data.data, "shape") and len(data.data.shape) == 2:
            is_latent_input = True

        X_explain = data.data

        if is_latent_input:
            # Reconstruct images for explanation
            with torch.no_grad():
                latent_tensor = torch.tensor(data.data, dtype=torch.float32).to(
                    self.device
                )
                recon_images = []
                bs = 100
                for i in range(0, len(latent_tensor), bs):
                    batch = latent_tensor[i : i + bs]
                    recon = self.autoencoder.decoder(batch)
                    recon_images.append(recon.cpu().numpy())
                X_explain = np.concatenate(recon_images, axis=0)

        # Get predictions for targets
        if is_latent_input:
            preds = model.predict_proba(data.data)  # model expects latent
        else:
            preds = model.predict_proba(X_explain)  # model expects images

        current_classes = np.argmax(preds, axis=1)
        targets = (current_classes + 1) % preds.shape[1]

        # Explain (works on images)
        explanation = self.explainer.explain(
            X=X_explain, Y_t=targets, batch_size=self.batch_size
        )

        if explanation.cf is not None and "X" in explanation.cf:
            cf_X = explanation.cf["X"]  # This is Images

            # Debug: check if cf_X rows are distinct
            if len(cf_X) > 1:
                diffs = np.abs(cf_X[0] - cf_X[1]).sum()
                print(
                    f"CFRLCelebA Debug: Difference between first two CF images: {diffs}"
                )
                unique_cfs = np.unique(cf_X.reshape(len(cf_X), -1), axis=0)
                print(
                    f"CFRLCelebA Debug: Number of unique CFs generated: {len(unique_cfs)} out of {len(cf_X)}"
                )

            # If input was latent, we must return latent counterfactuals
            if is_latent_input:
                with torch.no_grad():
                    cf_tensor = torch.tensor(cf_X, dtype=torch.float32).to(self.device)
                    # Encode back to latent
                    # Note: We encode the *counterfactual image* to get the z_cf that produced it (or close to it)
                    # Ideally, z_cf is what the Actor outputted.
                    # But we don't have access to Actor's internal z output easily here.
                    # Encoding the output image is consistent with the manifold assumption.
                    encoded_cfs = []
                    bs = 100
                    for i in range(0, len(cf_tensor), bs):
                        batch = cf_tensor[i : i + bs]
                        enc = self.autoencoder.encoder(batch)
                        encoded_cfs.append(enc.cpu().numpy())
                    cf_output = np.concatenate(encoded_cfs, axis=0)
            else:
                cf_output = cf_X  # Return images

            orig_data = data.data  # Return original as provided (latent or image)

            for i in range(len(data.data)):
                instance_cf = cf_output[i]
                instance_orig = orig_data[i]

                # Verify class
                if is_latent_input:
                    cf_prob = model.predict_proba(instance_cf[np.newaxis, ...])
                else:
                    cf_prob = model.predict_proba(instance_cf[np.newaxis, ...])

                cf_class = np.argmax(cf_prob)

                cfs.append(
                    Counterfactual(
                        original_data=instance_orig,
                        data=instance_cf,
                        original_class=int(current_classes[i]),
                        target_class=int(cf_class),
                        explainer=repr(self),
                    )
                )

        return cfs

    def __repr__(self) -> str:
        return f"cfrl_celeba(latent_dim={self.latent_dim}, coeff_sparsity={self.coeff_sparsity}, coeff_consistency={self.coeff_consistency}, train_steps={self.train_steps}, batch_size={self.batch_size})"


# Register Celery tasks
explainer = CFRLCelebA()
create_celery_tasks(explainer, "alibi_cfrl_celeba")
