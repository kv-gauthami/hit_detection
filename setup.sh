#!/usr/bin/env bash
# One-time setup: installs Python dependencies, clones TrackNetV3 and
# downloads its pretrained weights into third_party/TrackNetV3/ckpts/.
# Run from the repository root, inside an activated Python 3.10 env:
#     bash setup.sh
# PyTorch is installed automatically if missing (default build). An existing
# torch install is left untouched, so a specific CUDA build can be installed first.
set -euo pipefail

TN_DIR="third_party/TrackNetV3"
CKPT_DIR="$TN_DIR/ckpts"
GDRIVE_ID="1CfzE87a0f6LhBp0kniSl1-89zaLCZ8cA"   # checkpoint zip linked in TrackNetV3's README

echo "[0/4] Checking the Python environment"
PY_PREFIX=$(python -c 'import sys; print(sys.prefix)' 2>/dev/null || true)
ENV_PREFIX="${CONDA_PREFIX:-${VIRTUAL_ENV:-}}"
if [ -z "$ENV_PREFIX" ] || [ "$PY_PREFIX" != "$ENV_PREFIX" ]; then
  echo "  'python' ($PY_PREFIX) is not inside an activated conda/venv environment."
  echo "  Create one with Python included, e.g.:"
  echo "    conda create -n badminton python=3.10 -y && conda activate badminton"
  exit 1
fi
echo "  using $(python --version) at $PY_PREFIX"

echo "[1/4] Checking PyTorch"
if ! python -c "import torch" 2>/dev/null; then
  echo "  PyTorch not found, installing the default build."
  echo "  (For a specific CUDA version, install torch from https://pytorch.org first and rerun.)"
  python -m pip install torch torchvision
fi
python -c "import torch; print('  torch', torch.__version__, '| cuda:', torch.cuda.is_available())"

echo "[2/4] Installing Python requirements"
python -m pip install -r requirements.txt
python -m pip install gdown

echo "[3/4] Cloning TrackNetV3"
if [ -d "$TN_DIR/.git" ]; then
  echo "  already present, skipping"
else
  mkdir -p third_party
  git clone https://github.com/qaz812345/TrackNetV3.git "$TN_DIR"
fi

echo "[4/4] Downloading TrackNetV3 weights"
if [ -f "$CKPT_DIR/TrackNet_best.pt" ] && [ -f "$CKPT_DIR/InpaintNet_best.pt" ]; then
  echo "  already present, skipping"
else
  mkdir -p "$CKPT_DIR"
  TMP=$(mktemp -d)
  gdown "$GDRIVE_ID" -O "$TMP/ckpts.zip" \
    || gdown --fuzzy "https://drive.google.com/file/d/$GDRIVE_ID/view" -O "$TMP/ckpts.zip"
  unzip -q "$TMP/ckpts.zip" -d "$TMP/unzipped"
  # The zip layout may differ; find the two files wherever they landed.
  for f in TrackNet_best.pt InpaintNet_best.pt; do
    p=$(find "$TMP/unzipped" -name "$f" | head -1)
    [ -n "$p" ] || { echo "  $f not found in the downloaded zip"; exit 1; }
    mv "$p" "$CKPT_DIR/"
  done
  rm -rf "$TMP"
fi

ls -lh "$CKPT_DIR"
echo
echo "Setup complete. Try:"
echo "  python detect_hits.py path/to/clip.mp4 -o outputs/hits.csv"