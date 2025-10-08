from logging import warning
from explainers_lib.datasets import Dataset
from explainers_lib.model import Model
from celery import Celery
from typing import Set

# TODO: make this configurable
BROKER_URL = 'redis://localhost:6379/0'
BACKEND_URL = 'redis://localhost:6379/0'

app = Celery(
    'cf_ensemble',
    broker=BROKER_URL,
    backend=BACKEND_URL
)

app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
)

def try_get_available_explainers() -> Set[str]:
    i = app.control.inspect()
    try:
        active_queues_by_worker = i.active_queues()
        if active_queues_by_worker:
            all_queues = {
                queue['name']
                for queues in active_queues_by_worker.values()
                for queue in queues
            }
            return all_queues - {'celery'}
        else:
            return set()
    except (TimeoutError, Exception) as e:
        warning(f"Redis might be unreachable. Make sure to run:\ndocker run -d -p 6379:6379 --name celery-redis redis\n")
        return set()

@app.task(name='ensemble.collect_results')
def collect_results(results):
    return results

explainer_state = {}

def create_celery_tasks(explainer, name):
    explainer_state[name] = {
        'explainer': explainer
    }

    @app.task(name=f'{name}.repr')
    def get_repr(name=name):
        return repr(explainer_state[name]['explainer'])

    @app.task(name=f'{name}.set_dataset', ignore_result=True)
    def set_dataset(serialized_dataset: bytes, name=name):
        app.backend.client.set(f'explainer_data:{name}', serialized_dataset)

    @app.task(name=f'{name}.set_model', ignore_result=True)
    def set_model(serialized_model: bytes, model_type: str, name=name):
        app.backend.client.set(f'explainer_model:{name}', serialized_model)
        app.backend.client.set(f'explainer_model_type:{name}', model_type)

    @app.task(name=f'{name}.fit')
    def fit(_, name=name):
        serialized_dataset = app.backend.client.get(f'explainer_data:{name}')
        serialized_model = app.backend.client.get(f'explainer_model:{name}')
        model_type = app.backend.client.get(f'explainer_model_type:{name}').decode('utf-8')

        if not serialized_dataset or not serialized_model:
            raise RuntimeError(f"celery_remote: fit: data or model not set for {name}")

        data = Dataset.deserialize(serialized_dataset)
        model = Model.deserialize(serialized_model, model_type)

        explainer = explainer_state[name]['explainer']
        explainer.fit(model, data)

    @app.task(name=f'{name}.explain')
    def explain(_, name=name):
        serialized_dataset = app.backend.client.get(f'explainer_data:{name}')
        serialized_model = app.backend.client.get(f'explainer_model:{name}')
        model_type = app.backend.client.get(f'explainer_model_type:{name}').decode('utf-8')

        if not serialized_dataset or not serialized_model:
            raise RuntimeError(f"celery_remote: explain: data or model not set for {name}")

        data = Dataset.deserialize(serialized_dataset)
        model = Model.deserialize(serialized_model, model_type)

        explainer = explainer_state[name]['explainer']
        counterfactuals = explainer.explain(model, data)

        print(f"[DEBUG] Counterfactuals for {name}: {counterfactuals}")

        result = {
            'explainer': name,
            'instance': serialized_dataset,
            'counterfactuals': [counterfactual.serialize() for counterfactual in counterfactuals]
        }

        return result

    return get_repr, set_dataset, set_model, fit, explain
