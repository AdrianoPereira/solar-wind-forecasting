#!/usr/bin/env bash
# Build the submission image and run it the way Benchlab does, with the starter kit's mock upload.
#
#   bash submission/build.sh && bash submission/test_container.sh [T0]
#
# On a machine with Docker, the starter kit's local_tests/test_local.sh does the same with the
# real Dockerfile. This script is for rootless podman without subordinate UIDs (SDumont login
# nodes): it builds a local copy of the Dockerfile in which apt runs as root and the final switch
# to the `runner` user is dropped, because that user namespace has a single UID. The submitted
# Dockerfile is unchanged. CPU/memory limits are not applied (no cgroup controllers).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PACKAGE="${ROOT}/submission/dist/package"
MOCK="${ROOT}/benchlab-starter-kit/submission_pack/local_tests/mock_platform.py"
T0="${1:-$(date -u +%Y%m%dT%H0000Z)}"
PROXY="${HTTPS_PROXY:-}"
# podman's run root path must be short; keep the layers on local disk (NFS has no xattrs).
STORE="${PODMAN_STORE:-/tmp/${USER}-swp-podman}"
PM=(podman --root "${STORE}/root" --runroot "${STORE}/run" --storage-driver vfs
    --storage-opt vfs.ignore_chown_errors=true --cgroup-manager cgroupfs)
[ -d "${PACKAGE}" ] || { echo "run submission/build.sh first"; exit 1; }
[ -f "${MOCK}" ] || { echo "starter kit not found at ${MOCK}"; exit 1; }
mkdir -p "${STORE}"

LOCAL="${ROOT}/submission/dist/Dockerfile.local"
sed -e 's/apt-get update \&\& apt-get install/apt-get -o APT::Sandbox::User=root update \&\& apt-get -o APT::Sandbox::User=root install/' \
    -e '/useradd\|chown -R runner/d' -e '/^USER runner/d' \
    -e 's#^RUN chmod +x /app/run.sh /app/upload.sh \\#RUN chmod +x /app/run.sh /app/upload.sh#' \
    "${PACKAGE}/Dockerfile" > "${LOCAL}"
BUILD_ARGS=()
for name in http_proxy https_proxy HTTP_PROXY HTTPS_PROXY; do
    [ -n "${PROXY}" ] && BUILD_ARGS+=(--build-arg "${name}=${PROXY}")
done
"${PM[@]}" build --isolation oci --network host --platform linux/amd64 "${BUILD_ARGS[@]}" \
    -f "${LOCAL}" -t swp-local "${PACKAGE}"

SIZE=$("${PM[@]}" image inspect swp-local --format '{{.Size}}')
echo "image size : $((SIZE / 1024 / 1024)) MiB   (cap 15360 MiB)"

WORK="$(mktemp -d)"
trap 'kill ${MOCK_PID:-0} 2>/dev/null || true; rm -rf "${WORK}"' EXIT
(cd "${WORK}" && python3 "${MOCK}") & MOCK_PID=$!
sleep 1
START=$(date +%s)
"${PM[@]}" run --rm --network host \
    -e SUBMISSION_ID=sub_local -e INFERENCE_TIMESTAMP="${T0}" \
    -e HTTPS_PROXY="${PROXY}" -e HTTP_PROXY="${PROXY}" -e NO_PROXY=localhost,127.0.0.1 \
    -e OUTPUT_POST='{"url":"http://localhost:8899/","fields":{"key":"local"}}' \
    swp-local "${T0}"
echo "runtime    : $(( $(date +%s) - START ))s   (cap 900s INCLUDING pull)"

OUT="${WORK}/out/forecast.json"
[ -f "${OUT}" ] || { echo "FAIL: nothing uploaded"; exit 1; }
jq -e '.forecast_speed_kms | length == 72' "${OUT}" > /dev/null || { echo "FAIL: not 72 values"; exit 1; }
jq -e '[.forecast_speed_kms[] | select(. != null and (isnan | not))] | all(. >= 0 and . <= 10000)' "${OUT}" \
    > /dev/null || { echo "FAIL: value outside 0-10000 km/s"; exit 1; }
jq -c '{t0, valid_from, first: .forecast_speed_kms[0:3], last: .forecast_speed_kms[69:72]}' "${OUT}"
echo "PASS — forecast.json is valid"
