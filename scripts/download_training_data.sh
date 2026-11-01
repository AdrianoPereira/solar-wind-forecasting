#!/usr/bin/env bash
# Baixa em paralelo os dados de treino do modelo híbrido (buraco coronal + recorrência + L1):
#   - ACE Level 2 horário (AC_H2_SWE e AC_H2_MFI) do CDAWeb;
#   - ACE operacional horário (SWEPAM e MAG) em diretórios novos, sem tocar nos Parquet atuais;
#   - SDO/AIA 193 Å sinóptico a cada 6 h, um cliente por ano, no máximo AIA_JOBS ao mesmo tempo.
#
# Uso, na raiz do projeto (seguro para rodar de novo: tudo é retomado pelo cache):
#   nohup bash scripts/download_training_data.sh > data/download_logs/latest.out 2>&1 &
#
# Variáveis opcionais: START_YEAR (2011), END_DATE (ontem, UTC), AIA_JOBS (3),
# AIA_WORKERS (2), AIA_WAVELENGTH (193), CA_FILE (bundle do sistema, se existir),
# SKIP_AIA=1 ou SKIP_ACE=1 para pular um grupo.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

START_YEAR="${START_YEAR:-2011}"
END_DATE="${END_DATE:-$(date -u -d yesterday +%F)}"
AIA_JOBS="${AIA_JOBS:-3}"
AIA_WORKERS="${AIA_WORKERS:-2}"
AIA_WAVELENGTH="${AIA_WAVELENGTH:-193}"
CA_FILE="${CA_FILE:-/etc/pki/tls/certs/ca-bundle.crt}"
END_YEAR="${END_DATE:0:4}"

if [[ -x .venv/bin/python ]] && .venv/bin/python -c "import astropy, pyarrow, distributed" 2>/dev/null; then
    PY=(.venv/bin/python)
else
    PY=(uv run --locked --group images python)
fi
CA_ARGS=()
[[ -f "$CA_FILE" ]] && CA_ARGS=(--ca-file "$CA_FILE")

RUN="$(date -u +%Y%m%dT%H%M%SZ)"
LOGS="data/download_logs/$RUN"
mkdir -p "$LOGS"
STATUS="$LOGS/status.tsv"
: > "$STATUS"

log() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*"; }

# Roda um job em segundo plano, com log próprio, e registra código de saída e duração.
job() {
    local name="$1"; shift
    (
        local start=$SECONDS
        "$@" > "$LOGS/$name.log" 2>&1
        local code=$?
        printf '%s\t%s\t%ss\n' "$name" "$code" "$((SECONDS - start))" >> "$STATUS"
        log "fim $name (código $code, $((SECONDS - start))s)"
    ) &
    LAST_PID=$!
    log "início $name -> $LOGS/$name.log"
}

# Quantos dos PIDs informados ainda estão rodando.
running() {
    local count=0 pid
    for pid in "$@"; do
        kill -0 "$pid" 2>/dev/null && count=$((count + 1))
    done
    echo "$count"
}

log "execução $RUN: anos $START_YEAR–$END_YEAR até $END_DATE; python: ${PY[*]}"

if [[ "${SKIP_ACE:-0}" != 1 ]]; then
    job ace_level2 "${PY[@]}" scripts/download_ace_level2.py \
        --start "$START_YEAR-01-01" --end "$END_DATE" --workers 4 "${CA_ARGS[@]}"
    job ace_swepam_operational "${PY[@]}" scripts/download_ace_hourly.py \
        --start "$START_YEAR-01-01" --end "$END_DATE" --workers 6 \
        --output-dir "data/ace_hourly_${START_YEAR}_${END_YEAR}" "${CA_ARGS[@]}"
    job ace_mag_operational "${PY[@]}" scripts/download_ace_mag_hourly.py \
        --start "$START_YEAR-01-01" --end "$END_DATE" --workers 6 \
        --output-dir "data/ace_mag_hourly_${START_YEAR}_${END_YEAR}" "${CA_ARGS[@]}"
fi

if [[ "${SKIP_AIA:-0}" != 1 ]]; then
    # Anos diferentes gravam em dias diferentes; as gravações são atômicas e os
    # relatórios têm o período no nome, então os clientes não se sobrepõem.
    aia_pids=()
    for year in $(seq "$START_YEAR" "$END_YEAR"); do
        while (( $(running "${aia_pids[@]}") >= AIA_JOBS )); do
            wait -n
        done
        last="$year-12-31"
        [[ "$year" == "$END_YEAR" ]] && last="$END_DATE"
        job "aia_${AIA_WAVELENGTH}_$year" "${PY[@]}" scripts/download_aia_images.py \
            --start "$year-01-01" --end "$last" --wavelength "$AIA_WAVELENGTH" \
            --cadence-hours 6 --workers "$AIA_WORKERS" "${CA_ARGS[@]}"
        aia_pids+=("$LAST_PID")
    done
fi

wait
log "resumo (job, código, duração) em $STATUS:"
sort "$STATUS" | column -t
failed=$(awk -F'\t' '$2 != 0' "$STATUS" | wc -l)
if (( failed )); then
    log "$failed job(s) com falha ou cobertura incompleta; rode o mesmo comando de novo para retomar."
    exit 1
fi
log "todos os downloads terminaram sem falhas."
