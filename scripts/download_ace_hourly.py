#!/usr/bin/env python3
"""Baixa ACE/SWEPAM operacional horário desde 2020 e salva em Parquet.

Instalação: uv sync
Uso: uv run python scripts/download_ace_hourly.py
     uv run python scripts/download_ace_hourly.py --start 2020-01-01 --end 2020-01-31

Fonte: arquivo SolarSoft dos produtos NOAA/SWPC YYYYMM_ace_swepam_1h.txt.
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


def parse_month(content, month, filename):
    """Conserva velocidade original; coluna utilizável exige status nominal."""
    if "Hourly Averaged Real-time" not in content or "SWEPAM" not in content.upper().replace(
            "SOLAR WIND ELECTRON PROTON ALPHA MONITOR", "SWEPAM"):
        raise ValueError("Resposta não reconhecida como ACE/SWEPAM operacional horário")
    rows = []
    seen = set()
    next_month_start = (datetime(int(month[:4]), int(month[4:]), 1,
                                 tzinfo=timezone.utc) + timedelta(days=32)).replace(day=1)
    for line_number, line in enumerate(content.splitlines(), 1):
        line = line.strip()
        if not line or line.startswith(("#", ":")):
            continue
        parts = line.split()
        try:
            if len(parts) != 10:
                raise ValueError("esperadas 10 colunas")
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
            density, speed, temperature = map(float, parts[7:10])
            if status not in range(10) or not all(map(math.isfinite, (density, speed, temperature))):
                raise ValueError("status ou velocidade inválidos")
        except (ValueError, IndexError) as exc:
            raise ValueError(f"{filename}:{line_number}: {exc}") from exc
        seen.add(stamp)
        valid = status == 0 and speed >= 0
        rows.append({"timestamp_utc": stamp.strftime("%Y-%m-%dT%H:%M:%SZ"),
                     "speed_km_s_raw": speed, "status": status,
                     "is_valid": valid, "speed_km_s": speed if valid else None,
                     "density_cm3_raw": density,
                     "density_cm3": density if status == 0 and density >= 0 else None,
                     "temperature_k_raw": temperature,
                     "temperature_k": temperature if status == 0 and temperature >= 0 else None,
                     "source_file": filename})
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
    filename = f"{month}_ace_swepam_1h.txt"
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
    parser.add_argument("--output-dir", type=Path, default=Path("data/ace_hourly"))
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
                failure = {"month": month, "url": BASE_URL + f"{month}_ace_swepam_1h.txt",
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
    name = "ace_swepam_hourly_partial.parquet" if failures else "ace_swepam_hourly.parquet"
    schema = pa.schema([
        ("timestamp_utc", pa.timestamp("us", tz="UTC")),
        ("speed_km_s_raw", pa.float64()), ("status", pa.int8()),
        ("is_valid", pa.bool_()), ("speed_km_s", pa.float64()),
        ("density_cm3_raw", pa.float64()), ("density_cm3", pa.float64()),
        ("temperature_k_raw", pa.float64()), ("temperature_k", pa.float64()),
        ("source_file", pa.string()),
    ], metadata={b"product": b"ACE/SWEPAM NOAA/SWPC hourly real-time",
                 b"source": BASE_URL.encode(), b"speed_units": b"km/s",
                 b"density_units": b"protons/cm^3", b"temperature_units": b"K"})
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
    report = {"product": "ACE/SWEPAM NOAA/SWPC hourly real-time (SolarSoft archive)",
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
