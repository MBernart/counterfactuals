from explainers_lib.explainers.carla.growing_spheres import GrowingSpheresExplainer
from explainers_lib.explainers.celery_remote import app, create_celery_tasks

explainer = GrowingSpheresExplainer()

create_celery_tasks(explainer, 'carla_growing_spheres')
