from explainers_lib.explainers.dice.dice import DiceExplainer
from explainers_lib.explainers.celery_remote import app, create_celery_tasks

explainer = DiceExplainer()

create_celery_tasks(explainer, 'dice')