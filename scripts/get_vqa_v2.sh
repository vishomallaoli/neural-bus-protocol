#!/usr/bin/env bash
# Download COCO 2014 images + VQA v2 JSONs
# Usage:
#   bash scripts/get_vqa_v2.sh [ROOT=data] [SPLIT=val]
# SPLIT ∈ {val, train, both}
set -euo pipefail

ROOT="${1:-data}"
SPLIT="${2:-val}"
TMP="$ROOT/tmp"
COCO="$ROOT/coco"

mkdir -p "$TMP" "$COCO"

have() { command -v "$1" >/dev/null 2>&1; }
fetch() {
  local url="$1" out="$2"
  if [[ -f "$out" ]]; then
    echo "    ✓ Exists: $(basename "$out")"
    return
  fi
  echo "    ↓ Downloading $(basename "$out")"
  if have aria2c; then
    aria2c -x 8 -s 8 -o "$out" "$url"
  else
    curl -L -C - -o "$out" "$url"
  fi
}

unzip_if_needed() {
  local zip="$1" dest="$2"
  local marker="$dest/.unzipped_$(basename "$zip")"
  if [[ -f "$marker" ]]; then
    echo "    ✓ Already unzipped: $(basename "$zip")"
    return
  fi
  mkdir -p "$dest"
  echo "    ⇪ Unzipping $(basename "$zip") ..."
  unzip -q "$zip" -d "$dest"
  touch "$marker"
}

get_val() {
  echo "==> COCO 2014 val images ..."
  fetch "http://images.cocodataset.org/zips/val2014.zip" "$TMP/val2014.zip"
  unzip_if_needed "$TMP/val2014.zip" "$COCO"

  echo "==> VQA v2 JSONs (val) ..."
  fetch "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip"   "$TMP/v2_Questions_Val_mscoco.zip"
  fetch "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip" "$TMP/v2_Annotations_Val_mscoco.zip"
  unzip_if_needed "$TMP/v2_Questions_Val_mscoco.zip"   "$ROOT"
  unzip_if_needed "$TMP/v2_Annotations_Val_mscoco.zip" "$ROOT"
}

get_train() {
  echo "==> COCO 2014 train images ..."
  fetch "http://images.cocodataset.org/zips/train2014.zip" "$TMP/train2014.zip"
  unzip_if_needed "$TMP/train2014.zip" "$COCO"

  echo "==> VQA v2 JSONs (train) ..."
  fetch "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip"   "$TMP/v2_Questions_Train_mscoco.zip"
  fetch "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip" "$TMP/v2_Annotations_Train_mscoco.zip"
  unzip_if_needed "$TMP/v2_Questions_Train_mscoco.zip"   "$ROOT"
  unzip_if_needed "$TMP/v2_Annotations_Train_mscoco.zip" "$ROOT"
}

case "$SPLIT" in
  val)   get_val ;;
  train) get_train ;;
  both)  get_val; get_train ;;
  *)     echo "SPLIT must be one of {val, train, both}"; exit 1 ;;
esac

# Clean up temporary ZIP files
echo "==> Cleaning up temporary ZIP files ..."
rm -rf "$TMP"

echo "==> Done."
echo "   JSONs: $ROOT/v2_OpenEnded_mscoco_${SPLIT}2014_questions.json (or both)"
echo "          $ROOT/v2_mscoco_${SPLIT}2014_annotations.json (or both)"
echo "   Images: $COCO/{val2014,train2014}/COCO_*_2014_*.jpg"
