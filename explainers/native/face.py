from explainers_lib.explainers.native.face import FaceExplainer
from explainers_lib.explainers.celery_remote import app, create_celery_tasks

explainer = FaceExplainer()

create_celery_tasks(explainer, 'carla_face')