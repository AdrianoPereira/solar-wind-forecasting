#!/usr/bin/env python3
"""Baixa ACE/MAG operacional horário desde 2020 e salva em Parquet.

Instalação: uv sync
Uso: uv run python scripts/download_ace_mag_hourly.py
     uv run python scripts/download_ace_mag_hourly.py --start 2020-01-01 --end 2020-01-31

Fonte: arquivo SolarSoft dos produtos NOAA/SWPC YYYYMM_ace_mag_1h.txt.
Não usa Level 2. O produto-alvo exato do BenchLab ainda requer confirmação.
Arquivos mensais não garantem a versão disponível em cada instante histórico.
"""

import argparse
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import ssl
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

BASE_URL = "https://sohoftp.nascom.nasa.gov/sdb/goes/ace/monthly/"


def read_ca_file(path):
    # Pacotes do sistema podem conter comentários com caracteres não ASCII.
    content = path.read_text(encoding="utf-8")
    certificates = re.findall(
        r"-----BEGIN CERTIFICATE-----.*?-----END CERTIFICATE-----", content, re.DOTALL)
    if not certificates:
        raise ValueError("Nenhum certificado PEM encontrado")
    ca_data = "\n".join(certificates)
    ssl.create_default_context().load_verify_locations(cadata=ca_data)
    return ca_data


MAG_FIELDS = ("bx_gsm_nt", "by_gsm_nt", "bz_gsm_nt", "bt_nt",
              "latitude_gsm_deg", "longitude_gsm_deg")


def parse_month(content, month, filename):
    """Parse hourly GSM fields, preserving signed components and raw values."""
    if not all(marker in content for marker in (
            "Hourly Averaged Real-time Interplanetary Magnetic Field",
            "ACE Satellite - Magnetometer", "GSM coordinates")):
        raise ValueError("Resposta não reconhecida como ACE/MAG horário em GSM")
    rows, seen = [], set()
    next_month_start = (datetime(int(month[:4]), int(month[4:]), 1,
                                 tzinfo=timezone.utc) + timedelta(days=32)).replace(day=1)
    for line_number, line in enumerate(content.splitlines(), 1):
        line = line.strip()
        if not line or line.startswith(("#", ":")):
            continue
        parts = line.split()
        try:
            if len(parts) != 13:
                raise ValueError("esperadas 13 colunas")
            year, mon, day = map(int, parts[:3])
            hhmm = parts[3].zfill(4)
            stamp = datetime(year, mon, day, int(hhmm[:2]), int(hhmm[2:]),
                             tzinfo=timezone.utc)
            # Some monthly files include the first hour of the following month.
            if stamp == next_month_start:
                continue
            if stamp.strftime("%Y%m") != month or stamp.minute != 0:
                raise ValueError("timestamp fora do mês ou da grade horária")
            if stamp in seen:
                raise ValueError("timestamp duplicado")
            status = int(parts[6])
            values = list(map(float, parts[7:13]))
            if status not in range(10) or not all(map(math.isfinite, values)):
                raise ValueError("status ou campo magnético inválido")
        except (ValueError, IndexError) as exc:
            raise ValueError(f"{filename}:{line_number}: {exc}") from exc
        seen.add(stamp)
        row = {"timestamp_utc": stamp.strftime("%Y-%m-%dT%H:%M:%SZ"),
               "status": status, "source_file": filename}
        for field, value in zip(MAG_FIELDS, values):
            valid = status == 0 and value != -999.9
            if field == "bt_nt":
                valid = valid and value >= 0
            elif field == "latitude_gsm_deg":
                valid = valid and -90 <= value <= 90
            elif field == "longitude_gsm_deg":
                valid = valid and 0 <= value <= 360
            row[field + "_raw"] = value
            row[field] = value if valid else None
        # Field-vector validity does not require the angular columns.
        row["is_valid"] = all(row[field] is not None for field in MAG_FIELDS[:4])
        rows.append(row)
    if not rows:
        raise ValueError(f"{filename}: nenhum registro")
    return rows


def fetch(url, timeout, attempts, ca_data=None):
    context = ssl.create_default_context()
    if ca_data is not None:
        context.load_verify_locations(cadata=ca_data)
    for attempt in range(attempts):
        try:
            request = Request(url, headers={"User-Agent": "benchlab-ace-downloader/1.0"})
            with urlopen(request, timeout=timeout, context=context) as response:
                return response.read()
        except (URLError, TimeoutError, OSError) as exc:
            reason = getattr(exc, "reason", exc)
            if isinstance(reason, ssl.SSLCertVerificationError):
                raise ValueError(
                    "Falha na verificação SSL. Use --ca-file com um arquivo PEM de "
                    "CAs confiáveis do sistema ou fornecido pela instituição. "
                    f"Detalhe: {reason}"
                ) from exc
            if isinstance(exc, HTTPError) and exc.code not in (429, 500, 502, 503, 504):
                raise
            if attempt + 1 == attempts:
                raise
            time.sleep(min(2 ** attempt, 8))


def atomic_write(path, content):
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(content)
    temporary.replace(path)


def months(start, end):
    current = start.replace(day=1)
    while current <= end:
        yield current.strftime("%Y%m")
        current = (current.replace(day=28) + timedelta(days=4)).replace(day=1)


def download_month(month, raw_dir, refresh, timeout, attempts, ca_data=None):
    """Uma tarefa por mês; o cache deve ser compartilhado em clusters remotos."""
    filename = f"{month}_ace_mag_1h.txt"
    url = BASE_URL + filename
    path = raw_dir / filename
    metadata_path = path.with_suffix(".json")
    try:
        cached = False
        if path.exists() and metadata_path.exists() and not refresh:
            metadata = json.loads(metadata_path.read_text())
            # Reconsulta meses que ainda estavam abertos na última coleta.
            cached = metadata["retrieved_at"][:7].replace("-", "") > month
            cached = cached and metadata["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        if cached:
            payload = path.read_bytes()
        else:
            payload = fetch(url, timeout, attempts, ca_data)
        parsed = parse_month(payload.decode("utf-8-sig"), month, filename)
        if not cached:
            metadata = {"url": url, "retrieved_at": datetime.now(timezone.utc).isoformat(),
                        "sha256": hashlib.sha256(payload).hexdigest()}
            atomic_write(path, payload)
            atomic_write(metadata_path, json.dumps(metadata, indent=2).encode())
        return parsed, metadata, None, cached
    except (OSError, ValueError, KeyError) as exc:
        return [], None, {"month": month, "url": url, "error": str(exc)}, False


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", type=date.fromisoformat, default=date(2020, 1, 1))
    parser.add_argument("--end", type=date.fromisoformat,
                        default=datetime.now(timezone.utc).date(), help="Data final inclusiva (UTC)")
    parser.add_argument("--output-dir", type=Path, default=Path("data/ace_mag_hourly"))
    parser.add_argument("--timeout", type=float, default=45)
    parser.add_argument("--ca-file", type=Path,
                        help="Arquivo PEM com CAs confiáveis adicionais (enviado aos workers)")
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--refresh", action="store_true", help="Baixar novamente também os meses fechados")
    parser.add_argument("--workers", type=int, default=4, help="Downloads locais simultâneos (padrão: 4)")
    parser.add_argument("--scheduler", help="Endereço de um scheduler Dask remoto, ex.: tcp://host:8786")
    args = parser.parse_args()
    ca_data = None
    if args.ca_file is not None:
        try:
            ca_data = read_ca_file(args.ca_file)
        except (OSError, ValueError, ssl.SSLError) as exc:
            parser.error(f"Arquivo de CAs inválido: {exc}")
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        parser.error("Instale as dependências: uv sync")
    now = datetime.now(timezone.utc)
    if args.start > args.end or args.end > now.date():
        parser.error("Use start <= end <= hoje (UTC)")
    if args.timeout <= 0 or args.attempts < 1 or args.workers < 1:
        parser.error("timeout, attempts e workers devem ser positivos")
    output = args.output_dir
    raw_dir = output / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    records, files, failures = [], [], []
    from distributed import Client, as_completed

    # Um worker por download local; em cluster remoto vale a configuração do cluster.
    client_options = ({"address": args.scheduler} if args.scheduler else
                      {"n_workers": args.workers, "threads_per_worker": 1,
                       "processes": False, "dashboard_address": None})
    with Client(**client_options) as client:
        client.upload_file(__file__)
        futures = {
            client.submit(download_month, month, raw_dir.resolve(), args.refresh,
                          args.timeout, args.attempts, ca_data, pure=False): month
            for month in months(args.start, args.end)
        }
        for future in as_completed(futures):
            month = futures[future]
            try:
                parsed, metadata, failure, cached = future.result()
            except Exception as exc:
                parsed, metadata, cached = [], None, False
                failure = {"month": month, "url": BASE_URL + f"{month}_ace_mag_1h.txt",
                           "error": str(exc)}
            if failure:
                failures.append(failure)
                print(f"{month}: FALHA: {failure['error']}", file=sys.stderr)
                continue
            records.extend(row for row in parsed
                           if args.start.isoformat() <= row["timestamp_utc"][:10] <= args.end.isoformat()
                           and row["timestamp_utc"] <= now.strftime("%Y-%m-%dT%H:%M:%SZ"))
            files.append(metadata)
            print(f"{month}: {len(parsed)} registros ({'cache' if cached else 'download'})", file=sys.stderr)

    records.sort(key=lambda row: row["timestamp_utc"])
    # Resultados incompletos não sobrescrevem o arquivo de download completo.
    name = "ace_mag_hourly_partial.parquet" if failures else "ace_mag_hourly.parquet"
    schema = pa.schema([
        ("timestamp_utc", pa.timestamp("us", tz="UTC")),
        ("status", pa.int8()), ("is_valid", pa.bool_()),
        *[(name, pa.float64()) for field in MAG_FIELDS for name in (field + "_raw", field)],
        ("source_file", pa.string()),
    ], metadata={b"product": b"ACE/MAG NOAA/SWPC hourly real-time",
                 b"source": BASE_URL.encode(), b"coordinate_system": b"GSM",
                 b"magnetic_field_units": b"nT", b"angle_units": b"degrees"})
    table = pa.Table.from_pylist([
        dict(row, timestamp_utc=datetime.strptime(row["timestamp_utc"], "%Y-%m-%dT%H:%M:%SZ")
             .replace(tzinfo=timezone.utc)) for row in records
    ], schema=schema)
    temporary = output / (name + ".part")
    pq.write_table(table, temporary, compression="zstd")
    temporary.replace(output / name)
    start_hour = datetime.combine(args.start, datetime.min.time(), tzinfo=timezone.utc)
    end_exclusive = min(datetime.combine(args.end + timedelta(days=1), datetime.min.time(),
                                       tzinfo=timezone.utc), now.replace(minute=0, second=0, microsecond=0))
    present = {row["timestamp_utc"] for row in records}
    missing = []
    hour = start_hour
    while hour < end_exclusive:
        stamp = hour.strftime("%Y-%m-%dT%H:%M:%SZ")
        if stamp not in present:
            missing.append(stamp)
        hour += timedelta(hours=1)
    report = {"product": "ACE/MAG NOAA/SWPC hourly real-time (SolarSoft archive)",
              "coordinate_system": "GSM", "magnetic_field_units": "nT",
              "angle_units": "degrees",
              "validity_definition": "Nominal status and valid Bx, By, Bz, Bt; angles assessed separately",
              "start": args.start.isoformat(), "end_inclusive": args.end.isoformat(),
              "generated_at": now.isoformat(), "parquet": name,
              "rows": len(records), "valid_rows": sum(row["is_valid"] for row in records),
              "first_timestamp": records[0]["timestamp_utc"] if records else None,
              "last_timestamp": records[-1]["timestamp_utc"] if records else None,
              "missing_hours": missing, "failed_months": failures, "files": files,
              "note": "Sem imputação. Arquivo operacional não comprova disponibilidade histórica em t0."}
    atomic_write(output / "download_report.json", json.dumps(report, indent=2).encode())
    print(f"{output / name}: {len(records)} linhas; {len(missing)} horas ausentes; "
          f"{len(failures)} meses com falha.")
    return 1 if failures or not records else 0


if __name__ == "__main__":
    raise SystemExit(main())
