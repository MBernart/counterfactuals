import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import tensorflow as tf
tf.get_logger().setLevel(40) # suppress deprecation messages
tf.compat.v1.disable_v2_behavior() # disable TF2 behaviour as alibi code still relies on TF1 constructs

import os
import numpy as np
from alibi.explainers import CounterfactualProto

print('TF version: ', tf.__version__)
print('Eager execution enabled: ', tf.executing_eagerly()) # False

from explainers_lib.explainers.celery_remote import app, create_celery_tasks
from explainers_lib import Explainer, Dataset, Model, Counterfactual

class CFProto(Explainer):
    def fit(self, model: Model, data: Dataset) -> None:
        shape = (1,) + data.data.shape[1:]
        self.cf = CounterfactualProto(
            lambda x: model.predict_proba(x),
            shape,
            kappa=0.,
            beta=.1,
            gamma=100.,
            theta=100.,
            #  ae_model=ae, # TODO: figure out how to use autoencoders and categorical data
            #  enc_model=enc,
            max_iterations=500,
            feature_range=(-.5, .5), # TODO: take this from the dataset
            c_init=1.,
            c_steps=5,
            learning_rate_init=1e-2,
            clip=(-1000., 1000.))
        
        self.cf.fit(data.data)

    def explain(self, model: Model, data: Dataset) -> list[Counterfactual]:
        cfs = list()
        for instance in data.data:
            explanation = self.cf.explain(
                np.expand_dims(instance, axis=0),
                Y=None,
                target_class=None,
                k=20,
                k_type='mean',
                threshold=0.,
                verbose=True,
                print_every=100,
                log_every=100)
            if explanation.cf is not None:
                cfs.append(Counterfactual(
                    instance,
                    explanation.cf["X"][0],
                    explanation.orig_class,
                    explanation.cf["class"],
                    repr(self)
                ))
        return cfs

    def __repr__(self) -> str:
        return "cfproto()" # TODO: make the hyperparameters configurable

explainer = CFProto()
create_celery_tasks(explainer, "alibi_cfproto")
