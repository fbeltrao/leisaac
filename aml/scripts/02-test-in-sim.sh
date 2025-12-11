#!/bin/bash
set -e # Exit on error

echo "az ml job create --file aml/jobs/test-in-sim.yaml" "$@"
az ml job create --file aml/jobs/test-in-sim.yaml "$@"
