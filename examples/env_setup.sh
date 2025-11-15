#!/bin/bash

set -xe

docker pull redis:8.2.3
docker pull cfe.cs.put.poznan.pl/counterfactuals-native
docker pull cfe.cs.put.poznan.pl/counterfactuals-alibi
docker pull cfe.cs.put.poznan.pl/counterfactuals-carla
docker pull cfe.cs.put.poznan.pl/counterfactuals-dice

docker run --rm -d -p 6379:6379 --name celery-redis redis:8.2.3

# docker run --rm -d --network host --name counterfactuals-wachter cfe.cs.put.poznan.pl/counterfactuals-native wachter
# docker run --rm -d --network host --name counterfactuals-growing_spheres cfe.cs.put.poznan.pl/counterfactuals-native growing_spheres
docker run --rm -d --network host --name counterfactuals-face cfe.cs.put.poznan.pl/counterfactuals-native face

docker run --rm -d --network host --name counterfactuals-alibi-cfproto cfe.cs.put.poznan.pl/counterfactuals-alibi cfproto
docker run --rm -d --network host --name counterfactuals-alibi-cfrl cfe.cs.put.poznan.pl/counterfactuals-alibi cfrl

docker run --rm -d --network host --name counterfactuals-carla-actionable_recourse cfe.cs.put.poznan.pl/counterfactuals-carla actionable_recourse

docker run --rm -d --network host --name counterfactuals-dice cfe.cs.put.poznan.pl/counterfactuals-dice
