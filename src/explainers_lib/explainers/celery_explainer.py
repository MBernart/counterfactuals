from celery import chain, group, signature
from ..counterfactual import Counterfactual
from ..datasets import Dataset, Dataset
from ..model import Model
from . import Explainer
from .celery_remote import app
from typing import List

class CeleryExplainer(Explainer):
    """Celery Explainer"""

    def __init__(self, name: str):
        self.explainer_name = name

    def __repr__(self) -> str:
        return self.repr_async().apply_async().get()

    def repr_async(self) -> signature:
        return app.signature(
            f'{self.explainer_name}.repr',
            queue=self.explainer_name
        )

    def fit(self, model: Model, data: Dataset) -> None:
        """This method is used to fit the explainer"""

        fit_chain = self.fit_async(model, data)
        fit_chain.apply_async().get()

    def fit_async(self, model: Model, data: Dataset) -> chain:
        """This method is used to fit the explainer"""
        
        set_dataset_sig = app.signature(
            f'{self.explainer_name}.set_dataset',
            args=[data.serialize()],
            queue=self.explainer_name
        )

        set_model_sig = app.signature(
            f'{self.explainer_name}.set_model',
            args=[*model.serialize()],
            queue=self.explainer_name
        )

        fit_sig = app.signature(
            f'{self.explainer_name}.fit',
            queue=self.explainer_name
        )

        return group([set_dataset_sig, set_model_sig]) | fit_sig

    def explain(self, model: Model, data: Dataset) -> List[Counterfactual]:
        """This method is used generate the counterfactuals"""
        
        explain_chain = self.explain_async(model, data)
        result = explain_chain.apply_async().get()
        return [Counterfactual.deserialize(cf) for cf in result["counterfactuals"]]

    def explain_async(self, model: Model, data: Dataset) -> chain:
        """This method is used generate the counterfactuals"""
        
        set_dataset_sig = app.signature(
            f'{self.explainer_name}.set_dataset',
            args=[data.serialize()],
            queue=self.explainer_name
        )

        set_model_sig = app.signature(
            f'{self.explainer_name}.set_model',
            args=[*model.serialize()],
            queue=self.explainer_name
        )

        explain_sig = app.signature(
            f'{self.explainer_name}.explain',
            queue=self.explainer_name
        )

        return group([set_dataset_sig, set_model_sig]) | explain_sig

# Native explainers
class WachterExplainer(CeleryExplainer):
    def __init__(self):
        super().__init__("wachter")

class GrowingSpheresExplainer(CeleryExplainer):
    def __init__(self):
        super().__init__("growing_spheres")

class FaceExplainer(CeleryExplainer):
    def __init__(self):
        super().__init__("face")

# Dice explainer
class DiceExplainer(CeleryExplainer):
    def __init__(self):
        super().__init__("dice")

# Carla explainers
class ActionableRecourseExplainer(CeleryExplainer):
    def __init__(self):
        super().__init__("carla_actionable_recourse")

# Alibi Explain
class AlibiCFProto(CeleryExplainer):
    def __init__(self):
        super().__init__("alibi_cfproto")

class AlibiCFRL(CeleryExplainer):
    def __init__(self):
        super().__init__("alibi_cfrl")

class AlibiCFRLCelebA(CeleryExplainer):
    def __init__(self):
        super().__init__("alibi_cfrl_celeba")
