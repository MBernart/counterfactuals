from explainers_lib.datasets import SerializableDataset
from explainers_lib.model import TorchModel, TFModel
from explainers_lib.counterfactual import Counterfactual
from celery import Celery

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

@app.task(name='ensemble.get_explainers')
def get_explainers():
    return ['wachter', 'growing_spheres']

@app.task(name='ensemble.collect_results')
def collect_results(results):
    return results

explainer_state = {}

def create_celery_tasks(explainer, name):
    explainer_state[name] = {
        'explainer': explainer
    }

    @app.task(name=f'{name}.set_dataset', ignore_result=True)
    def set_dataset(serialized_dataset: bytes, name=name):
        app.backend.client.set(f'explainer_data:{name}', serialized_dataset)

    @app.task(name=f'{name}.set_model', ignore_result=True)
    def set_model(serialized_model: bytes, model_type: str, name=name):
        app.backend.client.set(f'explainer_model:{name}', serialized_model)
        app.backend.client.set(f'explainer_model_type:{name}', model_type)

    @app.task(name=f'{name}.fit', ignore_result=True)
    def fit(_, name=name):
        serialized_dataset = app.backend.client.get(f'explainer_data:{name}')
        serialized_model = app.backend.client.get(f'explainer_model:{name}')
        model_type = app.backend.client.get(f'explainer_model_type:{name}').decode('utf-8')

        if not serialized_dataset or not serialized_model:
            raise RuntimeError(f"celery_remote: fit: data or model not set for {name}")

        data = SerializableDataset.deserialize(serialized_dataset)
        if model_type == 'torch':
            model = TorchModel.deserialize(serialized_model)
        elif model_type == 'tf':
            model = TFModel.deserialize(serialized_model)
        else:
            raise NotImplementedError("celery_remote: set_model: Unknown model type")

        explainer = explainer_state[name]['explainer']
        explainer.fit(model, data)

    @app.task(name=f'{name}.explain')
    def explain(name=name):
        serialized_dataset = app.backend.client.get(f'explainer_data:{name}')
        serialized_model = app.backend.client.get(f'explainer_model:{name}')
        model_type = app.backend.client.get(f'explainer_model_type:{name}').decode('utf-8')

        if not serialized_dataset or not serialized_model:
            raise RuntimeError(f"celery_remote: explain: data or model not set for {name}")

        data = SerializableDataset.deserialize(serialized_dataset)
        if model_type == 'torch':
            model = TorchModel.deserialize(serialized_model)
        elif model_type == 'tf':
            model = TFModel.deserialize(serialized_model)
        else:
            raise NotImplementedError("celery_remote: set_model: Unknown model type")

        explainer = explainer_state[name]['explainer']
        counterfactuals = explainer.explain(model, data)

        print(f"[DEBUG] Counterfactuals for {name}: {counterfactuals}")

        result = {
            'explainer': name,
            'instance': serialized_dataset,
            'counterfactuals': [counterfactual.serialize() for counterfactual in counterfactuals]
        }

        print(f"[DEBUG] Result for {name}: {result}")

        return result

    return set_dataset, set_model, fit, explain
