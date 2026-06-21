#!/usr/bin/env bash
# git clone 후 한 번만 실행: bash setup.sh

set -e

ENV_NAME="${CONDA_ENV_NAME:-audiodream}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu118}"


if ! command -v conda &> /dev/null; then
    echo "[ERROR] conda not found. Please install Miniconda or Anaconda first."
    exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"

# 1. Conda environment
echo "[1/4] Preparing conda environment: ${ENV_NAME}"
if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "  conda env '${ENV_NAME}' already exists."
else
    conda env create -f environment.yml
fi

conda activate "${ENV_NAME}"

# 2. PyTorch with CUDA
echo "[2/4] Installing PyTorch CUDA wheels..."
python -m pip install --upgrade pip -q
python -m pip install torch torchvision torchaudio --index-url "${TORCH_INDEX_URL}" -q

# 3. Transformers from source (Qwen2-Audio requires recent support)
echo "[3/4] Installing transformers from source..."
python -m pip install git+https://github.com/huggingface/transformers -q

# 4. Other dependencies
echo "[4/4] Installing project dependencies..."
python -m pip install -r requirements.txt -q

echo ""
echo "=============================="
echo "  Setup complete!"
echo "  Activate: conda activate ${ENV_NAME}"
echo "  Open: http://localhost:7860"
echo "=============================="
