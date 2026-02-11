#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="path_foundation"
WITH_TRIDENT=0

print_usage() {
  cat <<'EOF'
Usage:
  bash create_env.sh [--name ENV_NAME] [--with-trident]

Options:
  --name ENV_NAME     Conda environment name (default: path_foundation)
  --with-trident      Install optional Trident/stain-normalisation extras
                      (Linux only): cupy-cuda12x, cucim, torch-staintools, TRIDENT
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --name)
      ENV_NAME="${2:-}"
      if [[ -z "${ENV_NAME}" ]]; then
        echo "Error: --name requires a value." >&2
        exit 1
      fi
      shift 2
      ;;
    --with-trident)
      WITH_TRIDENT=1
      shift
      ;;
    -h|--help)
      print_usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      print_usage
      exit 1
      ;;
  esac
done

if ! command -v conda >/dev/null 2>&1; then
  echo "Error: conda not found in PATH. Install Anaconda/Miniconda first." >&2
  exit 1
fi

OS_NAME="$(uname -s)"
ARCH_NAME="$(uname -m)"
echo "Detected OS: ${OS_NAME} (${ARCH_NAME})"
echo "Target env: ${ENV_NAME}"

echo "Creating base environment from environment.base.yml ..."
conda env create -n "${ENV_NAME}" -f environment.base.yml

if [[ "${OS_NAME}" == "Darwin" ]]; then
  echo "Applying macOS overlay (PyTorch/torchvision) ..."
  conda env update -n "${ENV_NAME}" -f environment.mac.yml

  if [[ "${ARCH_NAME}" == "arm64" ]]; then
    echo "Installing Apple Silicon TensorFlow packages ..."
    conda run -n "${ENV_NAME}" pip install tensorflow-macos tensorflow-metal
  else
    echo "Installing TensorFlow (CPU) for Intel macOS ..."
    conda run -n "${ENV_NAME}" pip install tensorflow
  fi

elif [[ "${OS_NAME}" == "Linux" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "NVIDIA GPU detected. Applying Linux CUDA overlay ..."
    conda env update -n "${ENV_NAME}" -f environment.linux-cuda.yml
  else
    echo "No NVIDIA GPU detected. Installing CPU PyTorch ..."
    conda install -n "${ENV_NAME}" -c pytorch pytorch torchvision cpuonly -y
  fi

  echo "Installing TensorFlow for Linux ..."
  conda run -n "${ENV_NAME}" pip install tensorflow

  if [[ "${WITH_TRIDENT}" -eq 1 ]]; then
    echo "Installing optional Trident/stain-normalisation packages ..."
    conda run -n "${ENV_NAME}" pip install cupy-cuda12x
    conda run -n "${ENV_NAME}" pip install cucim
    conda run -n "${ENV_NAME}" pip install torch-staintools
    conda run -n "${ENV_NAME}" pip install git+https://github.com/mahmoodlab/TRIDENT.git
  fi
else
  echo "Unsupported OS: ${OS_NAME}" >&2
  echo "Use macOS or Linux." >&2
  exit 1
fi

echo
echo "Environment setup complete."
echo "Activate with: conda activate ${ENV_NAME}"
