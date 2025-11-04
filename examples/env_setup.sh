#!/bin/bash

set -xe

docker pull redis:8.2.3
docker pull cfe.cs.put.poznan.pl/counterfactuals-alibi
docker pull cfe.cs.put.poznan.pl/counterfactuals-carla

docker run --rm -d -p 6379:6379 --name celery-redis redis:8.2.3

# celery -A explainers.wachter.main worker -l info -n wachter_worker@%h -Q wachter,celery
# celery -A explainers.growing_spheres.main worker -l info -n growing_spheres_worker@%h -Q growing_spheres,celery

docker run --rm -d --network host --name counterfactuals-alibi-cfproto cfe.cs.put.poznan.pl/counterfactuals-alibi cfproto
docker run --rm -d --network host --name counterfactuals-alibi-cfrl cfe.cs.put.poznan.pl/counterfactuals-alibi cfrl

docker run --rm -d --network host --name counterfactuals-carla-actionable_recourse cfe.cs.put.poznan.pl/counterfactuals-carla actionable_recourse
docker run --rm -d --network host --name counterfactuals-carla-dice cfe.cs.put.poznan.pl/counterfactuals-carla dice
docker run --rm -d --network host --name counterfactuals-carla-face cfe.cs.put.poznan.pl/counterfactuals-carla face