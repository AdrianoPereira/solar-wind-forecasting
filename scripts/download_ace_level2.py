#!/usr/bin/env python3
"""Baixa ACE Level 2 horário (SWEPAM e MAG) do CDAWeb via HAPI e salva em Parquet.

Uso: uv run python scripts/download_ace_level2.py --start 2011-01-01 --end 2025-12-31
     uv run python scripts/download_ace_level2.py --dataset AC_H2_SWE --workers 4

Fonte: https://cdaweb.gsfc.nasa.gov/hapi, conjuntos AC_H2_SWE e AC_H2_MFI.
É o produto científico, não o operacional usado como alvo ao vivo. Ele cobre
horas em que o arquivo operacional tem status não nominal; serve para treino
e auditoria, não comprova o que estava disponível em tempo real.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import date, datetime, timedelta, timezone
import hashlib
from io import StringIO
import json
import math
from pathlib import Path
import re
import ssl
import sys
import tempfile
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

BASE_URL = "https://cdaweb.gsfc.nasa.gov/hapi/"
FILL = -1.0e31
# Ordem dos parâmetros igual à do /info: o HAPI recusa outra ordem (erro 1411).
DATASETS = {
    "AC_H2_SWE": {
        "parameters": ["Np", "Vp", "Tpr", "alpha_ratio"],
        "columns": ["density_cm3", "speed_km_s", "temperature_k", "alpha_ratio"],
        "product": "ACE/SWEPAM Level 2 hourly (CDAWeb AC_H2_SWE)",
    },
    "AC_H2_MFI": {
        "parameters": ["Magnitude", "BGSM"],
        "columns": ["bt_nt", "bx_gsm_nt", "by_gsm_nt", "bz_gsm_nt"],
        "product": "ACE/MAG Level 2 hourly (CDAWeb AC_H2_MFI)",
    },
}
# Espaça o início das requisições entre threads para não sobrecarregar o servidor.
REQUEST_LOCK = threading.Lock()
NEXT_REQUEST = 0.0


def read_ca_file(path):
    content = path.read_text(encoding="utf-8")
    certificates = re.findall(
        r"-----BEGIN CERTIFICATE-----.*?-----END CERTIFICATE-----", content, re.DOTALL)
    if not certificates:
        raise ValueError("Nenhum certificado PEM encontrado")
    return "\n".join(certificates)


def ssl_context(ca_data):
    context = ssl.create_default_context()
    if ca_data:
        context.load_verify_locations(cadata=ca_data)
    return context


def atomic_write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".",
                                     suffix=".part", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def fetch(url, timeout, attempts, context, spacing):
    global NEXT_REQUEST
    for attempt in range(attempts):
        with REQUEST_LOCK:
            wait = NEXT_REQUEST - time.monotonic()
            NEXT_REQUEST = max(NEXT_REQUEST, time.monotonic()) + spacing
        if wait > 0:
            time.sleep(wait)
        try:
            with urlopen(url, timeout=timeout, context=context) as response:
                return response.read()
        except HTTPError as exc:
            # Erros do cliente (exceto limite de taxa) não melhoram com nova tentativa.
            if exc.code < 500 and exc.code != 429 or attempt + 1 == attempts:
                raise
            delay = float(exc.headers.get("Retry-After") or 0) or 2 ** (attempt + 2)
        except (URLError, TimeoutError, ConnectionError) as exc:
            if isinstance(getattr(exc, "reason", None), ssl.SSLCertVerificationError) or attempt + 1 == attempts:
                raise
            delay = 2 ** (attempt + 2)
        print(f"Nova tentativa {attempt + 2}/{attempts} em {delay:.0f}s: {url}", file=sys.stderr)
        time.sleep(delay)


def months(start, end):
    current = start.replace(day=1)
    while current <= end:
        yield current
        current = (current.replace(day=28) + timedelta(days=4)).replace(day=1)


def parse_month(content, dataset, month):
    """Converte o CSV do HAPI; sentinela -1e31 vira nulo, valor original é preservado."""
    columns = DATASETS[dataset]["columns"]
    rows = []
    for line_number, parts in enumerate(csv.reader(StringIO(content)), 1):
        if not parts:
            continue
        if len(parts) != len(columns) + 1:
            raise ValueError(f"linha {line_number}: esperadas {len(columns) + 1} colunas; "
                             f"início da resposta: {content[:120]!r}")
        stamp = datetime.strptime(parts[0][:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
        if stamp.strftime("%Y%m") != month:
            continue
        row = {"timestamp_utc": stamp}
        for name, text in zip(columns, parts[1:]):
            value = float(text)
            row[name + "_raw"] = value
            row[name] = None if value <= FILL / 10 or not math.isfinite(value) else value
        rows.append(row)
    return rows


def download_month(dataset, month_start, stop, raw_dir, refresh, timeout, attempts, context, spacing):
    month = month_start.strftime("%Y%m")
    month_end = (month_start.replace(day=28) + timedelta(days=4)).replace(day=1)
    query = {"id": dataset, "parameters": ",".join(DATASETS[dataset]["parameters"]),
             "time.min": f"{month_start:%Y-%m-%d}T00:00:00Z",
             "time.max": f"{min(month_end, stop):%Y-%m-%d}T00:00:00Z", "format": "csv"}
    url = BASE_URL + "data?" + urlencode(query, safe=",:")
    path = raw_dir / dataset / f"{month}.csv"
    metadata_path = path.with_suffix(".json")
    try:
        cached = False
        if path.exists() and metadata_path.exists() and not refresh:
            metadata = json.loads(metadata_path.read_text())
            # Reconsulta meses que ainda estavam abertos ou incompletos na última coleta.
            cached = (metadata["retrieved_at"][:7].replace("-", "") > month and metadata["url"] == url
                      and metadata["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest())
        payload = path.read_bytes() if cached else fetch(url, timeout, attempts, context, spacing)
        rows = parse_month(payload.decode("utf-8"), dataset, month)
        if not cached:
            metadata = {"url": url, "retrieved_at": datetime.now(timezone.utc).isoformat(),
                        "sha256": hashlib.sha256(payload).hexdigest(), "rows": len(rows)}
            atomic_write(path, payload)
            atomic_write(metadata_path, json.dumps(metadata, indent=2).encode())
        return rows, metadata, None, cached
    except (OSError, ValueError, KeyError) as exc:
        return [], None, {"month": month, "url": url, "error": f"{type(exc).__name__}: {exc}"}, False


def dataset_stop(dataset, timeout, attempts, context):
    info = json.loads(fetch(BASE_URL + "info?id=" + dataset, timeout, attempts, context, 0))
    names = [p["name"] for p in info["parameters"][1:]]
    expected = DATASETS[dataset]["parameters"]
    if [n for n in names if n in expected] != expected:
        raise ValueError(f"{dataset}: parâmetros inesperados no servidor: {names}")
    return datetime.strptime(info["stopDate"][:10], "%Y-%m-%d").date() + timedelta(days=1)


def write_dataset(output, dataset, rows, failures, files, start, end, stop):
    import pyarrow as pa
    import pyarrow.parquet as pq

    columns = DATASETS[dataset]["columns"]
    rows.sort(key=lambda row: row["timestamp_utc"])
    name = f"{dataset.lower()}_hourly{'_partial' if failures else ''}.parquet"
    fields = [("timestamp_utc", pa.timestamp("us", tz="UTC"))]
    for column in columns:
        fields += [(column + "_raw", pa.float64()), (column, pa.float64())]
    table = pa.Table.from_pylist(rows, schema=pa.schema(fields, metadata={
        b"product": DATASETS[dataset]["product"].encode(), b"source": BASE_URL.encode(),
        b"fill_value": str(FILL).encode()}))
    temporary = output / (name + ".part")
    pq.write_table(table, temporary, compression="zstd")
    temporary.replace(output / name)
    valid = {c: sum(row[c] is not None for row in rows) for c in columns}
    report = {"product": DATASETS[dataset]["product"], "dataset": dataset,
              "start": start.isoformat(), "end_inclusive": end.isoformat(),
              "server_stop_exclusive": stop.isoformat(), "parquet": name,
              "generated_at": datetime.now(timezone.utc).isoformat(),
              "rows": len(rows), "valid_values": valid,
              "first_timestamp": rows[0]["timestamp_utc"].isoformat() if rows else None,
              "last_timestamp": rows[-1]["timestamp_utc"].isoformat() if rows else None,
              "failed_months": failures, "files": files,
              "note": "Produto científico Level 2; sem imputação. Não comprova disponibilidade em tempo real."}
    atomic_write(output / f"{dataset.lower()}_report.json", json.dumps(report, indent=2).encode())
    print(f"{output / name}: {len(rows)} linhas; válidos {valid}; {len(failures)} meses com falha.")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", type=date.fromisoformat, default=date(2011, 1, 1))
    parser.add_argument("--end", type=date.fromisoformat,
                        default=datetime.now(timezone.utc).date(), help="Data final inclusiva (UTC)")
    parser.add_argument("--dataset", choices=sorted(DATASETS), action="append",
                        help="Repita para escolher; padrão: todos")
    parser.add_argument("--output-dir", type=Path, default=Path("data/ace_level2"))
    parser.add_argument("--workers", type=int, default=4, help="Downloads simultâneos (padrão: 4)")
    parser.add_argument("--spacing", type=float, default=0.5,
                        help="Intervalo mínimo entre inícios de requisições, em segundos")
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--attempts", type=int, default=5)
    parser.add_argument("--ca-file", type=Path)
    parser.add_argument("--refresh", action="store_true", help="Baixar novamente também os meses fechados")
    args = parser.parse_args()
    if args.start > args.end or args.end > datetime.now(timezone.utc).date():
        parser.error("Use start <= end <= hoje (UTC)")
    if args.timeout <= 0 or args.attempts < 1 or args.workers < 1 or args.spacing < 0:
        parser.error("timeout, attempts e workers devem ser positivos")
    try:
        import pyarrow  # noqa: F401
        context = ssl_context(read_ca_file(args.ca_file) if args.ca_file else None)
    except ImportError:
        parser.error("Instale as dependências: uv sync")
    except (OSError, ValueError, ssl.SSLError) as exc:
        parser.error(f"Arquivo de CAs inválido: {exc}")

    status = 0
    for dataset in args.dataset or sorted(DATASETS):
        try:
            stop = dataset_stop(dataset, args.timeout, args.attempts, context)
        except (OSError, ValueError, KeyError) as exc:
            print(f"{dataset}: FALHA ao ler /info: {exc}", file=sys.stderr)
            status = 1
            continue
        end = min(args.end, stop - timedelta(days=1))
        if args.start > end:
            print(f"{dataset}: o servidor só tem dados até {stop - timedelta(days=1)}", file=sys.stderr)
            continue
        stop_at = min(stop, end + timedelta(days=1))
        rows, files, failures = [], [], []
        with ThreadPoolExecutor(args.workers) as pool:
            futures = {pool.submit(download_month, dataset, month, stop_at, args.output_dir / "raw",
                                   args.refresh, args.timeout, args.attempts, context, args.spacing): month
                       for month in months(args.start, end)}
            for future in as_completed(futures):
                parsed, metadata, failure, cached = future.result()
                label = f"{dataset} {futures[future]:%Y-%m}"
                if failure:
                    failures.append(failure)
                    print(f"{label}: FALHA: {failure['error']}", file=sys.stderr)
                    continue
                rows.extend(row for row in parsed if args.start <= row["timestamp_utc"].date() <= end)
                files.append(metadata)
                print(f"{label}: {len(parsed)} registros ({'cache' if cached else 'download'})", file=sys.stderr)
        failures.sort(key=lambda failure: failure["month"])
        write_dataset(args.output_dir, dataset, rows, failures, files, args.start, end, stop)
        status = status or (1 if failures or not rows else 0)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
