#!/bin/bash
# OpenPI + RoboCasa environment setup
# Usage: bash setup.sh
# Creates conda env "openpi" with all dependencies, downloads kitchen assets.
# Does NOT download training datasets (see the Datasets section of README.md, or run: bash download_datasets.sh).
set -euo pipefail

ENV_NAME="openpi"
ROBOCASA_DIR="${HOME}/robocasa"

echo "=== Creating conda env: ${ENV_NAME} ==="
conda create -n "${ENV_NAME}" python=3.11 -y
eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"

echo "=== Installing openpi ==="
# Install the local openpi-client FIRST: openpi depends on "openpi-client", and a same-named
# package on PyPI (0.1.x, pinned to numpy<2) would otherwise be pulled in and conflict with the numpy pin below.
pip install -e packages/openpi-client/
pip install -e .

echo "=== Installing robocasa from source ==="
if [ ! -d "${ROBOCASA_DIR}" ]; then
    git clone https://github.com/robocasa/robocasa.git "${ROBOCASA_DIR}"
fi
pip install -e "${ROBOCASA_DIR}"

echo "=== Installing robosuite from master ==="
pip install git+https://github.com/ARISE-Initiative/robosuite.git@master

echo "=== Pinning numpy ==="
pip install numpy==2.2.5

echo "=== Downloading kitchen assets (~23 GB on disk after extraction) ==="
python -c "
from robocasa.scripts.download_kitchen_assets import download_and_extract_zip, DOWNLOAD_ASSET_REGISTRY
for name, config in DOWNLOAD_ASSET_REGISTRY.items():
    print(f'Downloading {name}...')
    download_and_extract_zip(**config)
print('Done.')
"

echo ""
echo "=== Setup complete ==="
echo "Activate with: conda activate ${ENV_NAME}"
echo "To download training data, run: bash download_datasets.sh (see the Datasets section of README.md)."
