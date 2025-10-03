from explainers_lib.explainers.wachter import WachterExplainer
from src.explainers_lib.explainers.celery_remote import create_celery_tasks

explainer = WachterExplainer()

create_celery_tasks(explainer, 'wachter')
