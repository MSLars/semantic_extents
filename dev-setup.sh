#!/bin/bash

export CONDA_ENV="reex"

# Setup conda env using mamba if installed
if type "mamba" >/dev/null; then
  export CONDA_CMD="mamba"
elif type "conda" >/dev/null; then
  export CONDA_CMD="conda"
else
  echo "No conda command found"
  exit 1
fi

# check if the conda env already existis. If so do only perform an update else create the env using the env.yml
if { conda env list | grep "$CONDA_ENV"; } >/dev/null 2>&1; then
  $CONDA_CMD env update -f env.yml
else
  $CONDA_CMD env create -f env.yml
fi

conda activate "$CONDA_ENV"
pip install -r requirements.txt
conda develop .

CUDA_AVAILABLE=$(printf 'import torch\nprint(torch.cuda.is_available())' | python)
if [ "$CUDA_AVAILABLE" = "True" ]; then
  echo "Environment setup successful and CUDA available"
else
  echo "Environment setup successful but CUDA not available"
  echo "Please ensure you have CUDA installed and re-run the setup"
  exit 1
fi
