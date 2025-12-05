import numpy as np
import torch
import torch.nn as nn
import os
from alibi.explainers import CounterfactualRL
from explainers_lib.explainers.celery_remote import app, create_celery_tasks
from explainers_lib import Explainer, Dataset, Model, Counterfactual

# --- Autoencoder and CNN Architectures from experiments/MNIST_ensemble.ipynb ---

import torch
import torch.nn as nn


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
    def __init__(self, latent_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBlock(1, 64),  # 32 -> 16
            ConvBlock(64, 128),  # 16 -> 8
            ConvBlock(128, 256),  # 8 -> 4
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, latent_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()

        self.fc = nn.Sequential(nn.Linear(latent_dim, 256 * 4 * 4), nn.Tanh())

        self.decoder = nn.Sequential(
            TransConvBlock(256, 128),  # 4 -> 8
            TransConvBlock(128, 64),  # 8 -> 16
            TransConvBlock(64, 32),  # 16 -> 32
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 256, 4, 4)
        x = self.decoder(x)
        return x


class Autoencoder(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = Encoder(latent_dim).to(device)
        self.decoder = Decoder(latent_dim).to(device)

    def forward(self, x):
        return self.decoder(self.encoder(x))


# --- CFRL Wrapper for MNIST ---


class CFRLMNIST(Explainer):
    def __init__(
        self,
        latent_dim=32,
        coeff_sparsity=15.0,
        coeff_consistency=0.5,
        train_steps=50_000,
        batch_size=128,
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
            "experiments/torch_ae_mnist_paper.pth",
            "../../experiments/torch_ae_mnist_paper.pth",
            "/home/berni/education/counterfactuals/experiments/models/torch_ae_mnist_paper.pth",
        ]

        ae_path = None
        for p in possible_paths:
            if os.path.exists(p):
                ae_path = p
                break

        if ae_path is None:
            raise FileNotFoundError(
                "Could not find 'torch_ae_mnist_paper.pth' in expected locations."
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
                "CFRLMNIST: Detected 2D input (latent). Reconstructing images for training..."
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
                    f"CFRLMNIST: Reconstructed {len(X_train)} images with shape {X_train.shape[1:]}."
                )

            # Predictor Wrapper for Latent Input Model:
            # Image -> Encoder -> Latent -> User Model

            predictor_wrapper = self._latent_predictor_wrapper
        else:
            # Standard Image Input
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

        if os.path.exists("cfrl_mnist.dill"):
            self.explainer.load("cfrl_mnist.dill", predictor=predictor_wrapper)
        else:
            self.explainer.fit(X=X_train)
            self.explainer.save("cfrl_mnist.dill")
        # self.save("cfrl/cfrl_mnist.pth")

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
                    f"CFRLMNIST Debug: Difference between first two CF images: {diffs}"
                )
                unique_cfs = np.unique(cf_X.reshape(len(cf_X), -1), axis=0)
                print(
                    f"CFRLMNIST Debug: Number of unique CFs generated: {len(unique_cfs)} out of {len(cf_X)}"
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
        return f"cfrl_MNIST(latent_dim={self.latent_dim}, coeff_sparsity={self.coeff_sparsity}, coeff_consistency={self.coeff_consistency}, train_steps={self.train_steps}, batch_size={self.batch_size})"

    def save(self, path: str) -> None:
        """
        Saves weights, params, and the preprocessor.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)

        state = {
            "params": {
                "latent_dim": self.latent_dim,
                "coeff_sparsity": self.coeff_sparsity,
                "coeff_consistency": self.coeff_consistency,
                "train_steps": self.train_steps,
                "batch_size": self.batch_size,
            },
            # Save Autoencoder Weights
            "autoencoder_state_dict": (
                self.autoencoder.state_dict() if self.autoencoder else None
            ),
            # Save Alibi Actor (The Counterfactual Generator)
            "actor_state_dict": (
                self.explainer.actor.state_dict() if self.explainer else None
            ),
            # CRITICAL: Save the Scikit-Learn Preprocessor
            # We must use pickle for this part, but Sklearn is usually safe to pickle.
            # If we don't save this, we won't know how to normalize input data later.
            "preprocessor": (
                self.preprocessor_pipeline
                if hasattr(self, "preprocessor_pipeline")
                else None
            ),
        }

        torch.save(state, path)
        print(f"CFRLCelebA saved to {path}")

    def load(self, path: str, model: Model) -> None:
        """
        Restores the explainer.
        REQUIRES 'model' because we must re-define the predictor function to build the object.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found at {path}")

        # 1. Load to CPU to avoid GPU conflicts
        state = torch.load(path, map_location=self.device)

        # 2. Restore Hyperparameters
        p = state["params"]
        self.latent_dim = p["latent_dim"]
        self.coeff_sparsity = p["coeff_sparsity"]
        self.coeff_consistency = p["coeff_consistency"]
        self.train_steps = p["train_steps"]
        self.batch_size = p["batch_size"]

        # 3. Restore Preprocessor (if you saved it)
        # self.preprocessor_pipeline = state.get("preprocessor")

        # 4. Re-initialize Autoencoder
        if state["autoencoder_state_dict"]:
            self.autoencoder = Autoencoder(latent_dim=self.latent_dim).to(self.device)
            self.autoencoder.load_state_dict(state["autoencoder_state_dict"])
            self.autoencoder.eval()

        # 5. Define the Wrapper (Re-create logic)
        # We need to detect if we should use latent or regular wrapper based on the loaded config
        # For safety, let's assume the regular wrapper or verify data shape if possible.
        # Here we define the standard one for reconstruction:

        def predictor_wrapper(x):
            # Same logic as your original _predictor_wrapper
            # If you are using the latent wrapper, you might need to handle that logic here
            return model.predict_proba(x)

        # 6. RE-INSTANTIATE the Alibi Explainer
        # We create a "fresh" explainer using the loaded hyperparams and restored Autoencoder
        self.explainer = CounterfactualRL(
            predictor=predictor_wrapper,
            encoder=self.autoencoder.encoder,
            decoder=self.autoencoder.decoder,
            latent_dim=self.latent_dim,
            coeff_sparsity=self.coeff_sparsity,
            coeff_consistency=self.coeff_consistency,
            train_steps=self.train_steps,  # Doesn't matter, we won't train
            batch_size=self.batch_size,
            backend="pytorch",
        )

        # 7. LOAD the Learned Policy (Actor)
        if state["actor_state_dict"]:
            # Now self.explainer exists, so we can load weights into it
            self.explainer.actor.load_state_dict(state["actor_state_dict"])
            print("CFRLCelebA: Actor weights loaded successfully.")
        else:
            print("Warning: No actor weights found in checkpoint.")


# Register Celery tasks
explainer = CFRLMNIST()
create_celery_tasks(explainer, "alibi_cfrl_mnist")
