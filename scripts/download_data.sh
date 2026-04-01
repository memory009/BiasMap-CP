#!/bin/bash
# Download all datasets for BiasMap-CP
# Usage: bash scripts/download_data.sh
set -e

DATA_ROOT="/LOCAL/psqhe8/BiasMap-CP/data/raw"
echo "Downloading datasets to $DATA_ROOT"

# ============================================================
# 1. VSR — Visual Spatial Reasoning
# ============================================================
echo "\n=== VSR ==="
VSR_DIR="$DATA_ROOT/vsr"
if [ ! -d "$VSR_DIR/.git" ]; then
    git clone https://github.com/cambridgeltl/visual-spatial-reasoning "$VSR_DIR"
else
    echo "VSR already cloned, pulling..."
    git -C "$VSR_DIR" pull
fi
# Download VSR images (they are MS-COCO + Visual Genome subset)
# Images are NOT in the repo — they reference COCO images
# Download COCO val2017 (5k images, ~800MB, sufficient for VSR test set)
COCO_IMG_DIR="$DATA_ROOT/coco/images/val2017"
if [ ! -d "$COCO_IMG_DIR" ] || [ "$(ls -A $COCO_IMG_DIR | wc -l)" -lt 1000 ]; then
    mkdir -p "$DATA_ROOT/coco/images"
    echo "Downloading COCO val2017..."
    wget -q --show-progress -O "$DATA_ROOT/coco/val2017.zip" \
        http://images.cocodataset.org/zips/val2017.zip
    unzip -q "$DATA_ROOT/coco/val2017.zip" -d "$DATA_ROOT/coco/images/"
    rm "$DATA_ROOT/coco/val2017.zip"
    echo "COCO val2017 downloaded."
else
    echo "COCO val2017 already exists."
fi
# Symlink for VSR image access
mkdir -p "$VSR_DIR/images"
[ -L "$VSR_DIR/images/val2017" ] || ln -sfn "$COCO_IMG_DIR" "$VSR_DIR/images/val2017"

# ============================================================
# 2. What'sUp
# ============================================================
echo "\n=== What'sUp ==="
WHATSUP_DIR="$DATA_ROOT/whatsup"
if [ ! -d "$WHATSUP_DIR/.git" ]; then
    git clone https://github.com/amitakamath/whatsup_vlms "$WHATSUP_DIR"
else
    git -C "$WHATSUP_DIR" pull
fi

# ============================================================
# 3. GSR-Bench
# ============================================================
echo "\n=== GSR-Bench ==="
GSR_DIR="$DATA_ROOT/gsr_bench"
if [ ! -d "$GSR_DIR/.git" ]; then
    git clone https://github.com/WenqiRajabi/GSR-BENCH "$GSR_DIR"
else
    git -C "$GSR_DIR" pull
fi

# ============================================================
# 4. GQA (questions + scene graphs only, images from huggingface)
# ============================================================
echo "\n=== GQA ==="
GQA_DIR="$DATA_ROOT/gqa"
mkdir -p "$GQA_DIR"
if [ ! -f "$GQA_DIR/questions1.2_train_balanced.json" ]; then
    echo "Downloading GQA questions..."
    wget -q --show-progress -O "$GQA_DIR/questions1.2.zip" \
        https://downloads.cs.stanford.edu/nlp/data/gqa/questions1.2.zip
    unzip -q "$GQA_DIR/questions1.2.zip" -d "$GQA_DIR/"
    rm "$GQA_DIR/questions1.2.zip"
fi
if [ ! -f "$GQA_DIR/sceneGraphs/train_sceneGraphs.json" ]; then
    echo "Downloading GQA scene graphs..."
    wget -q --show-progress -O "$GQA_DIR/sceneGraphs.zip" \
        https://downloads.cs.stanford.edu/nlp/data/gqa/sceneGraphs.zip
    unzip -q "$GQA_DIR/sceneGraphs.zip" -d "$GQA_DIR/"
    rm "$GQA_DIR/sceneGraphs.zip"
fi
# GQA images (~20GB) — download a subset using HuggingFace datasets
echo "Downloading GQA images via HuggingFace (this may take a while)..."
/LOCAL2/psqhe8/anaconda3/envs/llmav/bin/python - <<'EOF'
import os
from datasets import load_dataset
gqa_img_dir = "/LOCAL/psqhe8/BiasMap-CP/data/raw/gqa/images"
os.makedirs(gqa_img_dir, exist_ok=True)
# Load a sample to get images
ds = load_dataset("HuggingFaceM4/GQA", split="validation", streaming=True,
                  cache_dir="/LOCAL2/psqhe8/huggingface_cache/hub")
count = 0
for item in ds:
    img = item.get("image")
    img_id = item.get("id", str(count))
    if img:
        img.save(os.path.join(gqa_img_dir, f"{img_id}.jpg"))
    count += 1
    if count >= 5000:
        break
print(f"Saved {count} GQA images.")
EOF

# ============================================================
# 5. CLEVR
# ============================================================
echo "\n=== CLEVR ==="
CLEVR_DIR="$DATA_ROOT/clevr"
CLEVR_ZIP="$CLEVR_DIR/CLEVR_v1.0.zip"
if [ ! -d "$CLEVR_DIR/CLEVR_v1.0" ]; then
    mkdir -p "$CLEVR_DIR"
    echo "Downloading CLEVR (~18GB)..."
    wget -q --show-progress -O "$CLEVR_ZIP" \
        https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip
    echo "Extracting CLEVR..."
    unzip -q "$CLEVR_ZIP" -d "$CLEVR_DIR/"
    rm "$CLEVR_ZIP"
else
    echo "CLEVR already exists."
fi

# ============================================================
# 6. NLVR2
# ============================================================
echo "\n=== NLVR2 ==="
NLVR2_DIR="$DATA_ROOT/nlvr2"
mkdir -p "$NLVR2_DIR"
if [ ! -f "$NLVR2_DIR/train.jsonl" ]; then
    echo "Downloading NLVR2 annotations..."
    /LOCAL2/psqhe8/anaconda3/envs/llmav/bin/python - <<'EOF'
import os
from datasets import load_dataset
nlvr2_dir = "/LOCAL/psqhe8/BiasMap-CP/data/raw/nlvr2"
img_dir = os.path.join(nlvr2_dir, "images")
os.makedirs(img_dir, exist_ok=True)
ds = load_dataset("nlvr2", split="validation",
                  cache_dir="/LOCAL2/psqhe8/huggingface_cache/hub")
with open(os.path.join(nlvr2_dir, "dev.jsonl"), "w") as f:
    for item in ds:
        import json
        f.write(json.dumps({
            "identifier": item.get("identifier", ""),
            "sentence": item.get("sentence", ""),
            "label": item.get("label", "False"),
        }) + "\n")
        img = item.get("image_0") or item.get("image")
        if img:
            img.save(os.path.join(img_dir, f"{item.get('identifier','unk')}-img0.png"))
print("NLVR2 validation saved.")
EOF
fi

# ============================================================
# 7. SpatialSense (requires manual request to authors)
# ============================================================
echo "\n=== SpatialSense ==="
echo "NOTE: SpatialSense requires manual download from authors."
echo "  Request at: https://github.com/sled-group/SpatialSense"
echo "  Place the downloaded files in: $DATA_ROOT/spatialsense/"
echo "  The pipeline will skip SpatialSense if not available."

echo "\n=== Download Complete ==="
echo "Data root: $DATA_ROOT"
du -sh "$DATA_ROOT"/*/
