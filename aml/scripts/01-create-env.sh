#!/bin/bash
set -e # Exit on error

az ml environment create --name isaaclab-azcli-docker --build-context ./aml/docker/. --dockerfile-path Dockerfile --tags "git_hash=$(git rev-parse HEAD)" --description "Nvidia Isaac Lab with Azure CLI and Docker CLI" --version 2.3.1.2
