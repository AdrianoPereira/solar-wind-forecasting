#!/usr/bin/env bash
# Assemble the Benchlab submission package in submission/dist/ and zip it (Dockerfile at the ZIP root).
# Run from the repository root after scripts/export_submission_model.py:
#   bash submission/build.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${ROOT}/submission"
DIST="${SRC}/dist/package"
rm -rf "${SRC}/dist" && mkdir -p "${DIST}/benchlab_forecasting" "${DIST}/model"
cp "${SRC}"/{Dockerfile,README.md,forecast.py,requirements.txt,run.sh,submission.json,upload.sh} "${DIST}/"
# Only the inference modules of the package: the same code that built the training features.
cp "${ROOT}"/src/benchlab_forecasting/{__init__.py,hybrid.py,coronal_holes.py} "${DIST}/benchlab_forecasting/"
cp "${SRC}"/model/{tuned,tuned_no_ch}.{txt,json} "${DIST}/model/"
chmod +x "${DIST}"/{run.sh,upload.sh}
(cd "${DIST}" && zip -qr ../submission.zip .)
echo "package: ${DIST}"
echo "zip:     ${SRC}/dist/submission.zip ($(du -h "${SRC}/dist/submission.zip" | cut -f1))"
