from explainers_lib.explainers.native.growing_spheres import GrowingSpheresExplainer
from src.explainers_lib.explainers.celery_remote import app, create_celery_tasks

explainer = GrowingSpheresExplainer(step_size=0.1, max_radius=5.0, num_samples=1000)

create_celery_tasks(explainer, 'growing_spheres')