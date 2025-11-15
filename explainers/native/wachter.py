from explainers_lib.explainers.native.wachter import WachterExplainer
from src.explainers_lib.explainers.celery_remote import app, create_celery_tasks

explainer = WachterExplainer()

create_celery_tasks(explainer, 'wachter')
