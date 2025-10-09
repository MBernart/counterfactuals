from explainers_lib.explainers.actionable_recourse import ActionableRecourseExplainer
from explainers_lib.explainers.celery_remote import app, create_celery_tasks

explainer = ActionableRecourseExplainer()

create_celery_tasks(explainer, 'actionable_recourse')
