from explainers_lib.explainers.carla.actionable_recourse import ActionableRecourseExplainer
from explainers_lib.explainers.celery_remote import app, create_celery_tasks

explainer = ActionableRecourseExplainer()

create_celery_tasks(explainer, 'carla_actionable_recourse')
