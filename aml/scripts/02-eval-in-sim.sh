#!/bin/bash
set -e # Exit on error

# Parse training job and output name parameters
TRAINING_JOB=""
OUTPUT_NAME="checkpoint"
JOB_YAML_FILE="aml/jobs/eval-in-sim.yaml"
FILTERED_ARGS=()

while [[ $# -gt 0 ]]; do
  case $1 in
    --training-job)
      TRAINING_JOB="$2"
      shift 2
      ;;
    --output-name)
      OUTPUT_NAME="$2"
      shift 2
      ;;
    *)
      FILTERED_ARGS+=("$1")
      shift
      ;;
  esac
done

# Validate required parameters
if [[ -z "$TRAINING_JOB" ]]; then
  echo "Error: --training-job parameter is required"
  exit 1
fi


# Verify the output exists in the training job
echo "Verifying output '$OUTPUT_NAME' in training job '$TRAINING_JOB'..."
OUTPUT_EXISTS=$(az ml job show --name "$TRAINING_JOB" --query "outputs.${OUTPUT_NAME}" --output tsv)

if [[ -z "$OUTPUT_EXISTS" || "$OUTPUT_EXISTS" == "null" ]]; then
  echo "Error: Could not find output '$OUTPUT_NAME' in training job '$TRAINING_JOB'"
  exit 1
fi

# Construct the azureml data asset URI
DATA_ASSET_URI="azureml:azureml_${TRAINING_JOB}_output_data_${OUTPUT_NAME}:1"

# Add the checkpoint path to the filtered arguments
FILTERED_ARGS+=("--set" "inputs.checkpoint_path.path=$DATA_ASSET_URI")

echo "Resolved checkpoint path: $DATA_ASSET_URI"

printf 'az ml job create --file %s' "$JOB_YAML_FILE"
for arg in "${FILTERED_ARGS[@]}"; do
    if [[ $arg == *" "* ]]; then
        printf ' "%s"' "$arg"
    else
        printf ' %s' "$arg"
    fi
done
printf '\n'

az ml job create --file $JOB_YAML_FILE "${FILTERED_ARGS[@]}"
