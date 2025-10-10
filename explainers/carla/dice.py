from explainers_lib.explainers.carla.dice import DiceExplainer
from explainers_lib.explainers.celery_remote import app, create_celery_tasks

explainer = DiceExplainer()

create_celery_tasks(explainer, 'carla_dice')