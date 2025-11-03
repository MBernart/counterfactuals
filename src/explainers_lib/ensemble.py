from typing import Any, Callable, Optional, List, Dict
from celery import group, chain
from .model import Model
from .explainers import Explainer
from .explainers.celery_remote import app, try_get_available_explainers
from .explainers.celery_explainer import CeleryExplainer
from .aggregators import Aggregator, All, ScoreBasedAggregator
from .counterfactual import Counterfactual
from .datasets import Dataset
import numpy as np
from rich.console import Console
from rich.table import Table

Postprocessor = Callable[[np.ndarray], np.ndarray]

class Ensemble:
    def __init__(
        self, model: Model, explainers: List[Explainer], aggregator: Aggregator = All()
    ):
        """Constructs an ensemble"""
        self.model = model
        self.aggregator = aggregator

        self.explainers = list(filter(lambda explainer: not isinstance(explainer, CeleryExplainer), explainers))
        self.celery_explainers = list(filter(lambda explainer: isinstance(explainer, CeleryExplainer), explainers))
        self.celery_explainers = ensure_celery_explainers(self.celery_explainers)

    def get_explainers_repr(self) -> List[str]:
        task = repr_celery_explainers(self.celery_explainers)

        explainers = [repr(explainer) for explainer in self.explainers]

        if task:
            reprs = task.get()
            if isinstance(reprs, list):
                explainers.extend(reprs)
            elif isinstance(reprs, str):
                explainers.append(reprs)
            else:
                raise RuntimeError(f"Unkown explainers repr format: {reprs}")

        return explainers

    def fit(self, data: Dataset) -> None:
        """This method is used to train all explainers in the ensemble"""
        
        task = fit_celery_explainers(self.celery_explainers, self.model, data)

        for explainer in self.explainers:
            explainer.fit(self.model, data)

        if isinstance(self.aggregator, ScoreBasedAggregator):
            self.aggregator.fit(self.model, data)

        if task:
            task.get()

    def explain(self,
                data: Dataset,
                pretty_print: bool = False,
                pretty_print_postprocess: Optional[Postprocessor] = None,
                pretty_print_postprocess_target: Optional[Postprocessor] = None,
                feature_names: Optional[List[str]] = None) -> List[Counterfactual]:
        """This method is used to generate counterfactuals"""

        task = explain_celery_explainers(self.celery_explainers, self.model, data)

        all_counterfactuals = list()
        for explainer in self.explainers:
            cfs = explainer.explain(self.model, data)
            all_counterfactuals.extend(cfs)

        if task:
            results = task.get()
            if isinstance(results, list):
                cfs = [Counterfactual.deserialize(counterfactual) for result in results for counterfactual in result['counterfactuals']]
            elif isinstance(results, dict):
                cfs = [Counterfactual.deserialize(counterfactual) for counterfactual in results['counterfactuals']]
            else:
                raise RuntimeError(f"Unknown results format: {results}")
            all_counterfactuals.extend(cfs)

        all_filtered_counterfactuals = list()
        for cfs in cfs_group_by_original_data(all_counterfactuals).values():
            filtered_counterfactuals = self.aggregator(cfs)
            all_filtered_counterfactuals.extend(filtered_counterfactuals)
        
        if pretty_print:
            print_cfs(all_filtered_counterfactuals, data=data, model=self.model, explainers=self.get_explainers_repr(), postprocess=pretty_print_postprocess, postprocess_target=pretty_print_postprocess_target, feature_names=feature_names)

        return all_filtered_counterfactuals

def cfs_group_by_original_data(cfs: List[Counterfactual]) -> Dict[bytes, List[Counterfactual]]:
    table: Dict[bytes, List[Counterfactual]] = dict()

    for cf in cfs:
        key = cf.original_data.tobytes()
        if key in table:
            table[key].append(cf)
        else:
            table[key] = [cf]

    return table

def cfs_group_by_explainer(cfs: List[Counterfactual]) -> Dict[str, List[Counterfactual]]:
    table: Dict[str, List[Counterfactual]] = dict()

    for cf in cfs:
        key = cf.explainer
        if key in table:
            table[key].append(cf)
        else:
            table[key] = [cf]
    
    return table

def postprocess_cfs(cfs: List[Counterfactual],
                    postprocess: Postprocessor = lambda x: x,
                    postprocess_target: Postprocessor = lambda x: x) -> List[Counterfactual]:
    postprocessed_cfs = list()

    for cf in cfs:
        data = postprocess([cf.original_data, cf.data])
        targets = postprocess_target([cf.original_class, cf.target_class])

        postprocessed_cfs.append(Counterfactual(*data, *targets, cf.explainer))

    return postprocessed_cfs

def print_cfs(
        cfs: List[Counterfactual],
        feature_names: Optional[List[str]] = None,
        data: Optional[Dataset] = None,
        explainers: Optional[List[str]] = None,
        model: Optional[Model] = None,
        postprocess: Optional[Postprocessor] = None,
        postprocess_target: Optional[Postprocessor] = None,
        printer: Any = Console()) -> None:
    if len(cfs) == 0 and data is None:
        return
    
    feature_names = feature_names if feature_names else data.features if data else np.arange(cfs[0].original_data.shape[0]).tolist()

    table = Table()
    
    for feature_name in feature_names:
        table.add_column(feature_name, justify="right", no_wrap=True)
    table.add_column("target", justify="right", no_wrap=True)
    table.add_column("source")

    first_section = True

    by_instance = cfs_group_by_original_data(cfs)
    if data:
        for instance in data.data:
            key = instance.tobytes()
            if key not in by_instance:
                by_instance[key] = list()

    for bytes, cfs in by_instance.items():
        if not first_section:
            table.add_section()
        else:
            first_section = False

        original_data = cfs[0].original_data if len(cfs) > 0 else np.frombuffer(bytes)
        original_class = cfs[0].original_class if len(cfs) > 0 else model.predict(data.like(np.array([original_data])))[0] if model else None
        original_class = postprocess_target([original_class])[0] if postprocess_target else original_class
        original_class = repr(int(original_class)) if original_class is not None else "N/A" 
        original_data = postprocess([original_data])[0] if postprocess else original_data
        original_data = original_data.tolist()

        table.add_row(*map(lambda x: "{:.4f}".format(x) if isinstance(x, (int, float)) else str(x), original_data), original_class, "original data")


        by_explainer = cfs_group_by_explainer(cfs)
        if explainers:
            for explainer in explainers:
                if explainer not in by_explainer:
                    by_explainer[explainer] = list()

        for explainer, cfs in sorted(by_explainer.items(), key=lambda items: items[0]):
            if len(cfs) > 0:
                for cf in cfs:
                    cf_data = postprocess([cf.data])[0] if postprocess else cf.data
                    cf_data = cf_data.tolist()
                    cf_target = postprocess_target([cf.target_class])[0] if postprocess_target else cf.target_class
                    cf_target = repr(int(cf_target))
                    table.add_row(*map(lambda x: "{:.4f}".format(x) if isinstance(x, (int, float)) else str(x), cf_data), cf_target, cf.explainer)
            else:
                table.add_row(*["N/A" for _ in original_data], "N/A", explainer)
    
    printer.print(table)

def ensure_celery_explainers(requested_explainers: List[CeleryExplainer]) -> List[CeleryExplainer]:
    if len(requested_explainers) == 0:
        return []

    available_explainers = try_get_available_explainers()
    explainers_set = set(explainer.explainer_name for explainer in requested_explainers)

    explainers = list(filter(lambda explainer: explainer.explainer_name in available_explainers, requested_explainers))
    missing_explainers = list(explainers_set - available_explainers)

    if len(explainers) == 0:
        raise RuntimeError(f"Explainers not found: {missing_explainers}")
    
    if len(missing_explainers) > 0:
        raise RuntimeWarning(f"Explainers not found: {missing_explainers}")
    
    return explainers

def fit_celery_explainers(explainers: List[CeleryExplainer], model: Model, data: Dataset) -> Optional[Any]:
    if not explainers:
        return None

    fit_chains = [explainer.fit_async(model, data) for explainer in explainers]

    return chain(group(fit_chains), app.signature('ensemble.collect_results')).apply_async()


def explain_celery_explainers(explainers: List[CeleryExplainer], model: Model, data: Dataset) -> Optional[Any]:
    if not explainers:
        return None

    explain_chains = [explainer.explain_async(model, data) for explainer in explainers]

    return chain(group(explain_chains), app.signature('ensemble.collect_results')).apply_async()


def repr_celery_explainers(explainers: List[CeleryExplainer]) -> Optional[Any]:
    if not explainers:
        return None

    repr_group = [explainer.repr_async() for explainer in explainers]

    return chain(group(repr_group), app.signature('ensemble.collect_results')).apply_async()
