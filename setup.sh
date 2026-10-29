#!/usr/bin/env bash
# One-time setup: installs Python dependencies, clones TrackNetV3 and
# downloads its pretrained weights into third_party/TrackNetV3/ckpts/.
# Run from the repository root, inside an activated Python 3.10 env:
#     bash setup.sh
# Install PyTorch first (CUDA, Apple MPS or CPU build) from pytorch.org.
set -euo pipefail

TN_DIR="third_party/TrackNetV3"
CKPT_DIR="$TN_DIR/ckpts"
GDRIVE_ID="1CfzE87a0f6LhBp0kniSl1-89zaLCZ8cA"   # checkpoint zip linked in TrackNetV3's README

echo "[1/4] Checking PyTorch"
python -c "import torch; print('  torch', torch.__version__, '| cuda:', torch.cuda.is_available())" \
  || { echo "  PyTorch not found. Install it from https://pytorch.org first."; exit 1; }

echo "[2/4] Installing Python requirements"
pip install -r requirements.txt
pip install gdown

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