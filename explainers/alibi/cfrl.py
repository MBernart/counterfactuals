import numpy as np
from alibi.explainers import CounterfactualRLTabular
import pandas as pd
import tensorflow as tf
from explainers_lib.explainers.celery_remote import app, create_celery_tasks
from explainers_lib import Explainer, Dataset, Model, Counterfactual

class ListOutputModel(tf.keras.Model):
    """
    A wrapper to force Keras to return a list of outputs 
    even if there is only one output head.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def call(self, inputs, training=None, mask=None):
        outputs = self.model(inputs, training=training, mask=mask)
        # If Keras unwrapped the single output to a Tensor, wrap it back in a list
        if not isinstance(outputs, list):
            return [outputs]
        return outputs

    def get_config(self):
        return {"model": self.model}

class CFRL(Explainer):
    def __init__(self, latent_dim=8, coeff_sparsity=0.5, coeff_consistency=0.5, train_steps=1000, batch_size=10):
        self.coeff_sparsity = coeff_sparsity
        self.coeff_consistency = coeff_consistency
        self.train_steps = train_steps
        self.batch_size = batch_size
        self.latent_dim = latent_dim

    def fit(self, model: Model, data: Dataset) -> None:
        input_dim = data.data.shape[1]
        num_features_len = len(data.continuous_features)
        
        try:
            if 'cat' in data.preprocessor.named_transformers_:
                ohe_transformer = data.preprocessor.named_transformers_['cat'].named_steps['onehot']
                cat_output_dims = [len(cats) for cats in ohe_transformer.categories_]
            else:
                cat_output_dims = []
        except (KeyError, AttributeError):
            cat_output_dims = []

        encoder_input = tf.keras.layers.Input(shape=(input_dim,))
        
        encoder_hidden_dims = []
        current_dim = input_dim
        x = encoder_input

        # Dynamically create hidden layers
        while (current_dim // 2) > self.latent_dim:
            next_dim = current_dim // 2
            x = tf.keras.layers.Dense(next_dim, activation='relu')(x)
            encoder_hidden_dims.append(next_dim)
            current_dim = next_dim

        latent_space = tf.keras.layers.Dense(self.latent_dim, activation='relu', name='latent_space')(x)
        encoder = tf.keras.Model(encoder_input, latent_space, name="Encoder")

        decoder_input = tf.keras.layers.Input(shape=(self.latent_dim,))
        x = decoder_input

        for dim in reversed(encoder_hidden_dims):
            x = tf.keras.layers.Dense(dim, activation='relu')(x)

        decoder_outputs = []
        y_train_list = []
        losses = []
        
        if num_features_len > 0:
            output_num = tf.keras.layers.Dense(num_features_len, activation='linear', name='numerical_output')(x)
            decoder_outputs.append(output_num)
            y_train_list.append(data.data[:, :num_features_len])
            losses.append('mse')

        current_idx = num_features_len
        for i, dim in enumerate(cat_output_dims):
            out_cat = tf.keras.layers.Dense(dim, activation='softmax', name=f'categorical_output_{i}')(x)
            decoder_outputs.append(out_cat)
            y_train_list.append(data.data[:, current_idx : current_idx + dim])
            losses.append('categorical_crossentropy')
            current_idx += dim

        decoder = tf.keras.Model(decoder_input, decoder_outputs, name="Decoder")

        ae_input = tf.keras.layers.Input(shape=(input_dim,))
        ae_output = decoder(encoder(ae_input))
        autoencoder = tf.keras.Model(ae_input, ae_output)

        autoencoder.compile(optimizer='adam', loss=losses)
        
        autoencoder.fit(data.data, y_train_list, epochs=100, batch_size=16, verbose=True)

        if len(decoder_outputs) == 1:
            decoder_for_alibi = ListOutputModel(decoder)
        else:
            decoder_for_alibi = decoder

        def predictor_wrapper(x: pd.DataFrame) -> np.ndarray:
            preprocessed_x = data.preprocessor.transform(x)
            return model.predict_proba(preprocessed_x)

        if data.continuous_features:
            feature_ranges = {feat: [-1.0, 1.0] for feat in data.continuous_features}
        else:
            feature_ranges = {}

        if data.categorical_values:
            category_map = {data.features.index(feat): values for feat, values in data.categorical_values.items()}
        else:
            category_map = {}

        max_possible_steps = len(data.df) // self.batch_size
        self.train_steps = max(1, min(self.train_steps, max_possible_steps))

        self.explainer = CounterfactualRLTabular(
            predictor=predictor_wrapper,
            encoder=encoder,
            decoder=decoder_for_alibi,
            latent_dim=self.latent_dim,
            encoder_preprocessor=data.preprocessor.transform,
            decoder_inv_preprocessor=data.inverse_transform,
            coeff_sparsity=self.coeff_sparsity,
            coeff_consistency=self.coeff_consistency,
            category_map=category_map,
            feature_names=data.features,
            ranges=feature_ranges,
            immutable_features=data.immutable_features,
            train_steps=self.train_steps,
            batch_size=self.batch_size,
            backend="tensorflow")
        
        self.explainer.fit(X=data.df)

    def explain(self, model: Model, data: Dataset) -> list[Counterfactual]:
        cfs = list()
        all_targets = np.unique(data.target)
        
        if len(all_targets) < 2:
            print("Warning: Only one target class found. Cannot generate counterfactuals.")
            return []

        for i in range(len(data.df)):
            instance_df = data.df.iloc[i:i+1]
            original_target = data.target[i]

            possible_targets = all_targets[all_targets != original_target]
            if len(possible_targets) == 0:
                continue
            desired_target = possible_targets[0]
            
            Y_t = np.array([desired_target])

            explanation = self.explainer.explain(X=instance_df, Y_t=Y_t, C=[], patience=10000)
            
            if explanation.cf is not None:
                instance_orig_preprocessed = data.preprocessor.transform(explanation.orig['X']).ravel()
                instance_cf_preprocessed = data.preprocessor.transform(explanation.cf['X']).ravel()

                orig_class_raw = explanation.orig["class"]
                target_class_raw = explanation.cf["class"]

                if hasattr(orig_class_raw, 'flatten'):
                    c_orig = int(orig_class_raw.flatten()[0])
                else:
                    c_orig = int(orig_class_raw[0]) if isinstance(orig_class_raw, list) else int(orig_class_raw)

                if hasattr(target_class_raw, 'flatten'):
                    c_target = int(target_class_raw.flatten()[0])
                else:
                    c_target = int(target_class_raw[0]) if isinstance(target_class_raw, list) else int(target_class_raw)

                cfs.append(Counterfactual(
                    original_data=instance_orig_preprocessed,
                    data=instance_cf_preprocessed,
                    original_class=c_orig,
                    target_class=c_target,
                    explainer=repr(self)
                ))
        return cfs

    def __repr__(self) -> str:
        return f"cfrl(latent_dim={self.latent_dim}, coeff_sparsity={self.coeff_sparsity}, coeff_consistency={self.coeff_consistency}, train_steps={self.train_steps}, batch_size={self.batch_size})"

explainer = CFRL()
create_celery_tasks(explainer, "alibi_cfrl")
