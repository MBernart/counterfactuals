import keras
import numpy as np

from alibi.explainers import CounterfactualRLTabular
import tensorflow as tf

from explainers_lib.explainers.celery_remote import app, create_celery_tasks
from explainers_lib import Explainer, Dataset, Model, Counterfactual

class DecoderWrapper(keras.Model):
    """ Wraps a standard decoder to output a list with a single tensor. """
    def __init__(self, decoder, **kwargs):
        super().__init__(**kwargs)
        self.decoder = decoder
    def call(self, inputs):
        return [self.decoder(inputs)]

class CFRL(Explainer):
    def __init__(self, latent_dim=2, coeff_sparsity = 0.5, coeff_consistency = 0.5, train_steps = 1000, batch_size = 10):
        self.coeff_sparsity = coeff_sparsity
        self.coeff_consistency = coeff_consistency
        self.train_steps = train_steps
        self.batch_size = batch_size
        self.latent_dim = latent_dim

    def fit(self, model: Model, data: Dataset) -> None:
        # TODO: handle categorical data
        # TODO: more configurable autoencoders

        feature_types = {feature: float for feature in data.features}
        feature_ranges = (data.data.min(axis=0), data.data.max(axis=0))
        feature_ranges = {feature: [feature_ranges[0][i], feature_ranges[1][i]] for i, feature in enumerate(data.features)}

        input_dim = data.data.shape[1]

        encoder = tf.keras.models.Sequential([
            tf.keras.layers.InputLayer(input_shape=(input_dim,)),
            tf.keras.layers.Dense(8, activation='relu'),
            tf.keras.layers.Dense(self.latent_dim, activation='relu', name='latent_space')
        ])

        decoder = tf.keras.models.Sequential([
            tf.keras.layers.InputLayer(input_shape=(self.latent_dim,)),
            tf.keras.layers.Dense(8, activation='relu'),
            tf.keras.layers.Dense(input_dim, activation='sigmoid') # Sigmoid for outputs in [0, 1]
        ])

        autoencoder = tf.keras.models.Sequential([encoder, decoder])
        autoencoder.compile(optimizer='adam', loss='mse')

        autoencoder.fit(data.data, data.data, epochs=100, batch_size=16, verbose=True)

        max_possible_steps = len(data.data) // self.batch_size
        self.train_steps = max_possible_steps if self.train_steps > max_possible_steps else self.train_steps

        self.explainer = CounterfactualRLTabular(
            predictor=lambda x: model.predict_proba(data.like(x)),
            encoder=encoder,
            decoder=DecoderWrapper(decoder),
            latent_dim=self.latent_dim,
            encoder_preprocessor=lambda x: x,
            decoder_inv_preprocessor=lambda x: x,
            coeff_sparsity=self.coeff_sparsity,
            coeff_consistency=self.coeff_consistency,
            category_map={}, # TODO: get this from the dataset
            feature_names=data.features,
            ranges=feature_ranges,
            immutable_features=data.immutable_features,
            train_steps=self.train_steps,
            batch_size=self.batch_size,
            backend="tensorflow")
        
        self.explainer.fit(X=data.data)

    def explain(self, model: Model, data: Dataset) -> list[Counterfactual]:
        cfs = list()
        targets = set(data.target)
        for i, instance in enumerate(data.data):
            explanation = self.explainer.explain(
                np.expand_dims(instance, axis=0),
                np.array([[target for target in targets if target != data.target[i]][0]]), # TODO: this is a hack
                C=[]) # TODO: how to configure?
            if explanation.cf is not None:
                cfs.append(Counterfactual(
                    instance,
                    explanation.cf["X"][0],
                    explanation.orig["class"][0][0],
                    explanation.cf["class"][0][0], # TODO: this is also a hack
                    repr(self)
                ))
        return cfs

    def __repr__(self) -> str:
        return f"cfrl(latent_dim={self.latent_dim}, coeff_sparsity={self.coeff_sparsity}, coeff_consistency={self.coeff_consistency}, train_steps={self.train_steps}, batch_size={self.batch_size})"

explainer = CFRL()
create_celery_tasks(explainer, "alibi_cfrl")
