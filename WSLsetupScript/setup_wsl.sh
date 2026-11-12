#!/usr/bin/env bash
# Sets up WSL 2 (Ubuntu) for the 20 Newsgroups Tsetlin Machine scripts.
# Safe to run more than once: steps that are already done are skipped.
#
# Usage:
#   bash setup_wsl.sh                    # uses ~/TsetlinProject as the project folder
#   bash setup_wsl.sh ~/my/project       # or choose another folder
#
# Settings you can change when running it, for example:
#   HTM_SOURCE=github bash setup_wsl.sh  # install the library from GitHub instead of PyPI
#   INSTALL_TMU=1 bash setup_wsl.sh      # also install tmu (for the earlier, non-pair scripts)

set -euo pipefail

PROJECT_DIR="${1:-$HOME/TsetlinProject}"
HTM_SOURCE="${HTM_SOURCE:-pypi}"      # "pypi" (0.3.3, what you have now) or "github" (latest)
INSTALL_TMU="${INSTALL_TMU:-0}"
CUDA_KEYRING_URL="https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb"

step() { printf '\n\033[1;34m==> %s\033[0m\n' "$1"; }
ok()   { printf '\033[1;32m    %s\033[0m\n' "$1"; }
fail() { printf '\033[1;31mERROR: %s\033[0m\n' "$1" >&2; exit 1; }
# OUTPUT HELPERS
# Print each step in blue, successes in green and errors in red. set -euo pipefail stops the
# script at the first failing command, so it never carries on after a problem.

step "Checking that this is WSL with a visible NVIDIA GPU"
grep -qi microsoft /proc/version || fail "This does not look like WSL. Run the script inside your WSL Ubuntu terminal."
command -v nvidia-smi >/dev/null || fail "nvidia-smi not found. Update the NVIDIA driver in Windows, then run 'wsl --update' in PowerShell. Do not install a Linux NVIDIA driver inside WSL."
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || fail "nvidia-smi could not reach the GPU."
ok "GPU found"
# CHECK THE GPU
# WSL uses the Windows NVIDIA driver, so nvidia-smi must already work here. If it does not,
# the fix is on the Windows side, and installing anything in WSL will not help.

step "Installing build tools and Python"
sudo apt-get update -qq
sudo apt-get install -y -qq build-essential python3-dev python3-venv git wget
ok "Build tools ready ($(python3 --version))"
# BUILD TOOLS
# pycuda is compiled from source during installation, so it needs a C++ compiler
# (build-essential) and the Python headers (python3-dev).

step "Installing the CUDA Toolkit for WSL"
if command -v nvcc >/dev/null || [ -x /usr/local/cuda/bin/nvcc ]; then
    ok "CUDA Toolkit already installed, skipping"
else
    tmp_deb="$(mktemp --suffix=.deb)"
    wget -q -O "$tmp_deb" "$CUDA_KEYRING_URL"
    sudo dpkg -i "$tmp_deb"
    rm -f "$tmp_deb"
    sudo apt-get update -qq
    sudo apt-get install -y cuda-toolkit
    ok "CUDA Toolkit installed"
fi
# CUDA TOOLKIT
# Uses NVIDIA's WSL-Ubuntu package source, which provides the toolkit without a driver.
# The toolkit supplies cuda.h and nvcc, which pycuda needs to build and to compile GPU code.

step "Adding CUDA to PATH"
if ! grep -q '/usr/local/cuda/bin' "$HOME/.bashrc"; then
    # shellcheck disable=SC2016  # $PATH must stay unexpanded so .bashrc expands it later
    {
        echo ''
        echo '# CUDA (added by setup_wsl.sh)'
        echo 'export PATH=/usr/local/cuda/bin:$PATH'
        echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}'
    } >> "$HOME/.bashrc"
    ok "Added to ~/.bashrc"
else
    ok "Already in ~/.bashrc"
fi
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
nvcc --version | tail -n 2
# CUDA PATHS
# Adds CUDA to ~/.bashrc so new terminals find nvcc, and to this script's own environment so
# the next steps work straight away.

step "Creating the Python environment in $PROJECT_DIR"
mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR"
if [ ! -d .venv ]; then
    python3 -m venv .venv
    ok "Created .venv"
else
    ok ".venv already exists, reusing it"
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install -q --upgrade pip wheel setuptools
# VIRTUAL ENVIRONMENT
# Keeps the project's packages separate from the system Python. Keeping the project inside the
# Linux file system (not /mnt/c or /mnt/h) makes file access, including the spaCy cache, much faster.
case "$PROJECT_DIR" in
    /mnt/*) printf '\033[1;33m    Warning: %s is on the Windows drive, which is slow from WSL.\033[0m\n' "$PROJECT_DIR" ;;
esac

step "Installing Python packages"
python -m pip install -q numpy scipy scikit-learn scikit-image networkx spacy
python -m pip install -q pycuda
if [ "$HTM_SOURCE" = "github" ]; then
    python -m pip install -q "git+https://github.com/cair/PyHierarchicalTsetlinMachineCUDA.git"
else
    python -m pip install -q PyHierarchicalTsetlinMachineCUDA
fi
if [ "$INSTALL_TMU" = "1" ]; then
    python -m pip install -q "git+https://github.com/cair/tmu.git"
fi
ok "Packages installed"
# PYTHON PACKAGES
# - numpy, scipy, scikit-learn, scikit-image, networkx: the data tools and the library's dependencies.
# - spacy: tokenizing and lemmatizing the posts.
# - pycuda: lets Python run code on the GPU. It is installed before the Tsetlin Machine library.
# - PyHierarchicalTsetlinMachineCUDA: from PyPI (0.3.3) by default, the version your current
#   setup uses. HTM_SOURCE=github installs the latest version instead. The pair script works
#   with both.
# - tmu: only needed for the earlier scripts, so it is optional.

step "Downloading the spaCy English model"
if python -c "import spacy; spacy.load('en_core_web_sm')" 2>/dev/null; then
    ok "en_core_web_sm already installed"
else
    python -m spacy download en_core_web_sm
fi
# SPACY MODEL
# The model is a separate download from spaCy itself. Without it, the scripts fail with
# "Can't find model 'en_core_web_sm'".

step "Testing the GPU from Python"
python - <<'EOF'
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
print("    GPU:", cuda.Device(0).name())
mod = SourceModule("__global__ void twice(float *a) { a[threadIdx.x] *= 2; }")
a = np.arange(4, dtype=np.float32)
mod.get_function("twice")(cuda.InOut(a), block=(4, 1, 1), grid=(1, 1))
assert np.allclose(a, [0, 2, 4, 6]), a
print("    Compiled and ran a GPU kernel successfully")
from PyHierarchicalTsetlinMachineCUDA.tm import MultiClassTsetlinMachine
print("    PyHierarchicalTsetlinMachineCUDA imports correctly")
EOF
# GPU TEST
# Compiles and runs a tiny GPU program, the same thing the Tsetlin Machine library does when it
# starts. If this passes, pycuda, nvcc and the GPU all work together.

step "Done"
echo "    Project folder: $PROJECT_DIR"
echo "    To start working in a new terminal:"
echo "        cd $PROJECT_DIR && source .venv/bin/activate"
echo "    Then, for example:"
echo "        python main.py --categories sci.crypt rec.sport.hockey --epochs 5 --features 200"
