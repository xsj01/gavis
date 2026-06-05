#!/usr/bin/env bash
#
# Download the Barboursville "partial" dataset from Google Drive and extract it
# into data/Barboursville/partial/.
#
# The archive is laid out as Barboursville/partial/{depth,img,transforms.json,
# eval.json}, so it is extracted into data/ and lands directly at
# data/Barboursville/partial/.
#
# Usage:
#   bash scripts/download_data.sh
#
set -euo pipefail

FILE_ID="1DdUs2fO7_YVJ-X787aIdnm361TQwUcEc"

# Resolve paths relative to the repo root (this script lives in scripts/).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_DIR="${REPO_ROOT}/data"
DEST_DIR="${DATA_DIR}/Barboursville/partial"

# Pick whichever gdown is available (CLI or python module).
if command -v gdown >/dev/null 2>&1; then
    GDOWN=(gdown)
else
    GDOWN=(python -m gdown)
fi

ZIP_PATH="$(mktemp -t barboursville.XXXXXX.zip)"
trap 'rm -f "${ZIP_PATH}"' EXIT

echo "Downloading Barboursville dataset (Google Drive id ${FILE_ID})..."
"${GDOWN[@]}" "https://drive.google.com/uc?id=${FILE_ID}" -O "${ZIP_PATH}"

echo "Extracting into ${DATA_DIR}/ ..."
mkdir -p "${DATA_DIR}"
unzip -o -q "${ZIP_PATH}" -d "${DATA_DIR}"

echo "Done. Dataset available at ${DEST_DIR}"
