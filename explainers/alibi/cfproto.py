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

# note: This explainer does not support immutable features
class CFProto(Explainer):
    def __init__(self):
        self.cf = None
        self.safe_k = 1 # Fallback

    def fit(self, model: Model, data: Dataset) -> None:
        print(f"CFProto: Fitting on {len(data.data)} samples.")
        
        # CFProto builds KD-Trees based on y_pred, not y_true.
        preds = model.predict_proba(data.data)
        predicted_classes = np.argmax(preds, axis=1)
        unique_preds, pred_counts = np.unique(predicted_classes, return_counts=True)
        
        if len(pred_counts) > 0:
            min_pred_samples = np.min(pred_counts)
        else:
            min_pred_samples = 0

        self.safe_k = int(max(1, min(20, min_pred_samples)))
        
        print(f"CFProto: Model prediction distribution: {dict(zip(unique_preds, pred_counts))}")
        print(f"CFProto: Dynamic 'k' set to {self.safe_k} based on smallest predicted bucket.")

        num_transformer = data.preprocessor.named_transformers_['num']
        cat_transformer = data.preprocessor.named_transformers_['cat']
        onehot_encoder = cat_transformer.named_steps['onehot']

        if data.continuous_features:
            mins_orig = [data.allowable_ranges[feat][0] for feat in data.continuous_features]
            maxs_orig = [data.allowable_ranges[feat][1] for feat in data.continuous_features]
            
            scaled_ranges = num_transformer.transform(np.array([mins_orig, maxs_orig]))
            scaled_mins_num = scaled_ranges[0]
            scaled_maxs_num = scaled_ranges[1]
        else:
            scaled_mins_num = np.array([])
            scaled_maxs_num = np.array([])

        cat_vars_dict = {}
        if data.categorical_features:
            is_ohe = True
            categories_list = onehot_encoder.categories_
            
            current_index = len(scaled_mins_num) 
            
            all_cat_mins = []
            all_cat_maxs = []

            for categories in categories_list:
                num_categories = len(categories)
                
                if onehot_encoder.drop == 'if_binary' and num_categories == 2:
                    n_output_cols = 1
                elif onehot_encoder.drop == 'first' and num_categories > 0:
                    n_output_cols = num_categories - 1
                else: # Includes 'drop=None'
                    n_output_cols = num_categories

                cat_vars_dict[current_index] = num_categories
                
                all_cat_mins.append(0)
                all_cat_maxs.append(1)
                
                current_index += n_output_cols

            scaled_mins_cat = np.array(all_cat_mins)
            scaled_maxs_cat = np.array(all_cat_maxs)

        else:
            is_ohe = False
            scaled_mins_cat = np.array([])
            scaled_maxs_cat = np.array([])

        feature_range_mins = np.concatenate([scaled_mins_num, scaled_mins_cat])
        feature_range_maxs = np.concatenate([scaled_maxs_num, scaled_maxs_cat])
        
        feature_range_mins = feature_range_mins.reshape(1, -1)
        feature_range_maxs = feature_range_maxs.reshape(1, -1)

        shape = (1,) + data.data.shape[1:]
        self.cf = CounterfactualProto(
            lambda x: model.predict_proba(x),
            shape,
            kappa=0.,
            beta=.1, # Increase to heavily penalize changes.
            gamma=100.,
            theta=100.,
            max_iterations=500,
            feature_range=(feature_range_mins, feature_range_maxs),
            cat_vars=cat_vars_dict,
            ohe=is_ohe,
            use_kdtree=True,
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
                k=self.safe_k, 
                k_type='mean',
                threshold=0.,
                verbose=True,
                print_every=100,
                log_every=100)
            
            if explanation.cf is not None:
                orig_class_raw = explanation.orig_class
                target_class_raw = explanation.cf["class"]
                
                if hasattr(orig_class_raw, 'item'): c_orig = orig_class_raw.item()
                elif isinstance(orig_class_raw, np.ndarray): c_orig = int(orig_class_raw.flatten()[0])
                else: c_orig = int(orig_class_raw)

                if hasattr(target_class_raw, 'item'): c_target = target_class_raw.item()
                elif isinstance(target_class_raw, np.ndarray): c_target = int(target_class_raw.flatten()[0])
                else: c_target = int(target_class_raw)

                cfs.append(Counterfactual(
                    instance,
                    explanation.cf["X"][0],
                    c_orig,
                    c_target,
                    repr(self)
                ))
        return cfs

    def __repr__(self) -> str:
        return "cfproto()" # TODO: make the hyperparameters configurable

explainer = CFProto()
create_celery_tasks(explainer, "alibi_cfproto")
