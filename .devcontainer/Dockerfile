FROM mcr.microsoft.com/devcontainers/python:3.10

# Install Redis CLI for debugging
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends redis-tools

WORKDIR /workspaces/counterfactuals
