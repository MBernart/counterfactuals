from celery import Celery

# We use Redis as both the message broker and the result backend.
# The 'redis://redis:6379/0' URL assumes you're using Docker Compose,
# where 'redis' is the service name of the Redis container.
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
    timezone='Europe/Warsaw',
    enable_utc=True,
)

# Import the explainers to register the tasks
import explainers.wachter.main
import explainers.growing_spheres.main

@app.task(name='ensemble.get_explainers')
def get_explainers():
    return ['wachter', 'growing_spheres']

@app.task(name='ensemble.collect_results')
def collect_results(results):
    return results