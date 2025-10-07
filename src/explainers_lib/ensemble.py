from typing import Any, Optional
from celery import group
from .model import Model
from .explainers import Explainer
from .explainers.celery_remote import app, try_get_available_explainers
from .explainers.celery_explainer import CeleryExplainer
from .aggregators import Aggregator, All, Pareto
from .counterfactual import Counterfactual
from .datasets import Dataset
import numpy as np
from rich.console import Console
from rich.table import Table

class Ensemble:
    def __init__(
        self, model: Model, explainers: list[Explainer], aggregator: Aggregator = All()
    ):
        """Constructs an ensemble"""
        self.model = model
        self.aggregator = aggregator

        self.explainers = list(filter(lambda explainer: not isinstance(explainer, CeleryExplainer), explainers))
        self.celery_explainers = list(filter(lambda explainer: isinstance(explainer, CeleryExplainer), explainers))
        self.celery_explainers = ensure_celery_explainers(self.celery_explainers)

    def get_explainers_repr(self) -> str:
        task = repr_celery_explainers(self.celery_explainers)

        explainers = [repr(explainer) for explainer in self.explainers]

        if task:
            explainers.extend(task.get())

        return explainers

    def fit(self, data: Dataset) -> None:
        """This method is used to train all explainers in the ensemble"""
        
        task = fit_celery_explainers(self.celery_explainers, self.model, data)

        for explainer in self.explainers:
            explainer.fit(self.model, data)

        if isinstance(self.aggregator, Pareto):
            self.aggregator.fit(self.model, data)

        if task:
            task.get()

    def explain(self, data: Dataset, pretty_print: bool = False) -> list[Counterfactual]:
        """This method is used to generate counterfactuals"""

        task = explain_celery_explainers(self.celery_explainers, self.model, data)

        all_counterfactuals = list()
        for explainer in self.explainers:
            cfs = explainer.explain(self.model, data)
            all_counterfactuals.extend(cfs)

        if task:
            results = task.get()
            cfs = [Counterfactual.deserialize(counterfactual) for result in results for counterfactual in result['counterfactuals']]
            all_counterfactuals.extend(cfs)

        all_filtered_counterfactuals = list()
        for cfs in cfs_group_by_original_data(all_counterfactuals).values():
            filtered_counterfactuals = self.aggregator(cfs)
            all_filtered_counterfactuals.extend(filtered_counterfactuals)
        
        if pretty_print:
            print_cfs(all_filtered_counterfactuals, data=data, model=self.model, explainers=self.get_explainers_repr())

        return all_filtered_counterfactuals

def cfs_group_by_original_data(cfs: list[Counterfactual]) -> dict[bytes, list[Counterfactual]]:
    table: dict[bytes, list[Counterfactual]] = dict()

    for cf in cfs:
        key = cf.original_data.tobytes()
        if key in table:
            table[key].append(cf)
        else:
            table[key] = [cf]

    return table

def cfs_group_by_explainer(cfs: list[Counterfactual]) -> dict[str, list[Counterfactual]]:
    table: dict[str, list[Counterfactual]] = dict()

    for cf in cfs:
        key = cf.explainer
        if key in table:
            table[key].append(cf)
        else:
            table[key] = [cf]
    
    return table

def print_cfs(
        cfs: list[Counterfactual],
        feature_names: Optional[list[str]] = None,
        data: Optional[Dataset] = None,
        explainers: Optional[list[str]] = None,
        model: Optional[Model] = None,
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
        original_class = repr(int(cfs[0].original_class)) if len(cfs) > 0 else repr(int(model.predict(data.like(np.array([original_data])))[0])) if model else "N/A"
        original_data = original_data.tolist()

        table.add_row(*map(lambda x: "{:.4f}".format(x), original_data), original_class, "original data")

        by_explainer = cfs_group_by_explainer(cfs)
        if explainers:
            for explainer in explainers:
                if explainer not in by_explainer:
                    by_explainer[explainer] = list()

        for explainer, cfs in sorted(by_explainer.items(), key=lambda items: items[0]):
            if len(cfs) > 0:
                for cf in cfs:
                    table.add_row(*map(lambda x: "{:.4f}".format(x), cf.data.tolist()), repr(int(cf.target_class)), cf.explainer)
            else:
                table.add_row(*["N/A" for _ in original_data], "N/A", explainer)
    
    printer.print(table)

def ensure_celery_explainers(requested_explainers: list[CeleryExplainer]) -> list[CeleryExplainer]:
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

def fit_celery_explainers(explainers: list[CeleryExplainer], model: Model, data: Dataset) -> Any | None:
    if len(explainers) == 0:
        return None

    fit_chains = []
    for explainer in explainers:
        fit_chains.append(explainer.fit_async(model, data))

    return (group(fit_chains) | app.signature('ensemble.collect_results')).apply_async()

def explain_celery_explainers(explainers: list[CeleryExplainer], model: Model, data: Dataset) -> Any | None:
    if len(explainers) == 0:
        return None

    explain_chains = []
    for explainer in explainers:
        explain_chains.append(explainer.explain_async(model, data))

    return (group(explain_chains) | app.signature('ensemble.collect_results')).apply_async()

def repr_celery_explainers(explainers: list[CeleryExplainer]) -> Any | None:
    if len(explainers) == 0:
        return None

    repr_group = []
    for explainer in explainers:
        repr_group.append(explainer.repr_async())
    
    return (group(repr_group) | app.signature('ensemble.collect_results')).apply_async()
