#!/usr/bin/env python3
"""Download sampled SDO/AIA synoptic FITS images with provenance and coverage.

Example:
  uv run --group images python scripts/download_aia_images.py \
    --start 2024-03-01 --end 2024-03-02 --wavelength 193 --cadence-hours 6

Select the latest acceptable observation at or before each origin minus the
assumed latency. Historical retrieval does not establish real-time availability.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, time, timedelta, timezone
import hashlib
from html.parser import HTMLParser
from io import BytesIO
import json
import math
from pathlib import Path
import re
import sys
import tempfile
import ssl
import time as clock_time
import threading
from email.utils import parsedate_to_datetime
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from download_ace_hourly import read_ca_file
from aia_fits import open_aia_fits

BASE_URL = "https://jsoc1.stanford.edu/data/aia/synoptic/"
FILE_PATTERN = re.compile(r"AIA(\d{8})_(\d{4})_(\d{4})\.fits")
UTC = timezone.utc
REQUEST_LOCK = threading.Lock()
NEXT_REQUEST = 0.0


def fetch(url, timeout, attempts, ca_data=None):
    """Pace request starts across workers and back off on temporary failures."""
    global NEXT_REQUEST
    context = ssl.create_default_context()
    if ca_data is not None:
        context.load_verify_locations(cadata=ca_data)
    for attempt in range(attempts):
        with REQUEST_LOCK:
            wait = max(0.0, NEXT_REQUEST - clock_time.monotonic())
            if wait:
                clock_time.sleep(wait)
            NEXT_REQUEST = clock_time.monotonic() + 1.0
        try:
            request = Request(url, headers={"User-Agent": "benchlab-aia-downloader/1.1"})
            with urlopen(request, timeout=timeout, context=context) as response:
                return response.read()
        except (URLError, TimeoutError, OSError) as exc:
            if isinstance(getattr(exc, "reason", exc), ssl.SSLCertVerificationError):
                raise ValueError(f"SSL verification failed; use --ca-file with trusted CAs: {exc}") from exc
            if isinstance(exc, HTTPError) and exc.code not in (429, 500, 502, 503, 504):
                raise
            if attempt + 1 == attempts:
                raise
            delay = min(5 * 2 ** attempt, 60)
            if isinstance(exc, HTTPError) and exc.headers:
                retry_after = exc.headers.get("Retry-After")
                if retry_after:
                    try:
                        retry_seconds = float(retry_after)
                    except ValueError:
                        try:
                            retry_seconds = (parsedate_to_datetime(retry_after) - datetime.now(UTC)).total_seconds()
                        except (ValueError, TypeError, OverflowError):
                            retry_seconds = 0
                    if math.isfinite(retry_seconds):
                        delay = max(delay, retry_seconds)
            # Share the cooldown: another worker must not immediately retry.
            with REQUEST_LOCK:
                NEXT_REQUEST = max(NEXT_REQUEST, clock_time.monotonic() + delay)
            print(f"Retry {attempt + 2}/{attempts} in at least {delay:.0f}s: {url}: {exc}", file=sys.stderr)


def utc_string(stamp):
    return stamp.isoformat().replace("+00:00", "Z")


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


class FileLinks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.names = set()

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            href = dict(attrs).get("href", "")
            # Accept only plain basenames from the known archive layout.
            if FILE_PATTERN.fullmatch(href):
                self.names.add(href)


def parse_index(content, wavelength, earliest, cutoff):
    parser = FileLinks()
    parser.feed(content)
    found = []
    for name in parser.names:
        match = FILE_PATTERN.fullmatch(name)
        if int(match[3]) != wavelength:
            continue
        stamp = datetime.strptime(match[1] + match[2], "%Y%m%d%H%M").replace(tzinfo=UTC)
        # Minute-level filenames need a one-minute allowance at the lower edge.
        if earliest - timedelta(minutes=1) <= stamp <= cutoff:
            found.append((stamp, name))
    return sorted(found, reverse=True)


def inspect_fits(payload, wavelength):
    import numpy as np
    from astropy.time import Time

    with open_aia_fits(BytesIO(payload)) as (hdus, header_notes):
        hdu = next((h for h in hdus if h.data is not None and h.data.ndim == 2), None)
        if hdu is None or hdu.data.shape != (1024, 1024):
            raise ValueError("Expected a 1024 x 1024 synoptic image")
        header = hdu.header
        if not str(header.get("INSTRUME", "")).startswith("AIA"):
            raise ValueError("FITS instrument is not AIA")
        if int(header["WAVELNTH"]) != wavelength:
            raise ValueError("FITS wavelength does not match the request")
        quality = int(header["QUALITY"])
        exposure = float(header["EXPTIME"])
        if quality != 0 or not math.isfinite(exposure) or exposure <= 0:
            raise ValueError(f"Rejected quality/exposure: QUALITY={quality}, EXPTIME={exposure}")
        if not np.isfinite(hdu.data).any():
            raise ValueError("Image has no finite pixels")
        # AIA DATE-OBS is UTC unless a TIMESYS keyword explicitly states otherwise.
        scale = str(header.get("TIMESYS", "UTC")).strip().lower()
        if scale not in ("utc", "tai", "tt"):
            raise ValueError(f"Unsupported FITS time scale: {scale}")
        observed = Time(str(header["DATE-OBS"]).rstrip("Z"), format="isot", scale=scale).utc.to_datetime(timezone=UTC)
        metadata = {"observation_utc": utc_string(observed),
                    "header_normalizations": header_notes,
                    "date_obs_original": str(header["DATE-OBS"]), "time_scale": scale,
                    "time_scale_basis": "TIMESYS" if "TIMESYS" in header else "AIA DATE-OBS convention",
                    "wavelength_angstrom": wavelength, "quality": quality,
                    "exposure_seconds": exposure, "shape": list(hdu.data.shape),
                    "data_level": header.get("LVL_NUM"), "pixel_units": header.get("BUNIT"),
                    "instrument": header.get("INSTRUME"),
                    "finite_pixel_fraction": float(np.isfinite(hdu.data).mean()),
                    "wcs": {k: header.get(k) for k in ["CTYPE1", "CTYPE2", "CUNIT1", "CUNIT2",
                        "CRPIX1", "CRPIX2", "CDELT1", "CDELT2", "CRVAL1", "CRVAL2", "RSUN_OBS"]}}
        return observed, metadata


def load_image(url, path, args, ca_data):
    sidecar = path.with_suffix(".json")
    if not args.refresh and path.exists() and sidecar.exists():
        try:
            metadata = json.loads(sidecar.read_text())
            payload = path.read_bytes()
            if metadata["url"] == url and metadata["sha256"] == hashlib.sha256(payload).hexdigest():
                observed, fields = inspect_fits(payload, args.wavelength)
                metadata.update(fields)
                return observed, metadata, True
        except (OSError, ValueError, KeyError):
            pass
    payload = fetch(url, args.timeout, args.attempts, ca_data)
    observed, metadata = inspect_fits(payload, args.wavelength)
    metadata.update({"url": url, "retrieved_at_utc": utc_string(datetime.now(UTC)),
                     "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload),
                     "historical_publication_time_utc": None,
                     "product": "SDO/AIA synoptic FITS (archive)",
                     "note": "Archive processing level is recorded from LVL_NUM; real-time equivalence is unverified."})
    atomic_write(path, payload)
    atomic_write(sidecar, json.dumps(metadata, indent=2).encode())
    return observed, metadata, False


def download_slot(origin, args, ca_data):
    cutoff = origin - timedelta(minutes=args.latency_minutes)
    earliest = cutoff - timedelta(minutes=args.max_age_minutes)
    result = {"origin_utc": utc_string(origin), "cutoff_utc": utc_string(cutoff),
              "wavelength_angstrom": args.wavelength, "status": "missing", "errors": []}
    hour = cutoff.replace(minute=0, second=0, microsecond=0)
    while hour >= earliest.replace(minute=0, second=0, microsecond=0):
        directory = args.base_url.rstrip("/") + hour.strftime("/%Y/%m/%d/H%H00/")
        try:
            listing = fetch(directory, args.timeout, args.attempts, ca_data).decode("utf-8")
        except HTTPError as exc:
            if exc.code != 404:
                raise
            result["errors"].append({"url": directory, "error": "Directory not found (404)"})
            hour -= timedelta(hours=1)
            continue
        for _, name in parse_index(listing, args.wavelength, earliest, cutoff):
            path = args.output_dir / "raw" / hour.strftime("%Y/%m/%d") / name
            try:
                observed, metadata, cached = load_image(directory + name, path, args, ca_data)
                if not earliest <= observed <= cutoff:
                    continue
                result.update({"status": "ok", "file": str(path.relative_to(args.output_dir)),
                               "cached": cached, "age_at_origin_minutes": (origin - observed).total_seconds() / 60,
                               **metadata})
                return result
            except (OSError, ValueError, KeyError) as exc:
                result["errors"].append({"url": directory + name, "error": str(exc)})
        hour -= timedelta(hours=1)
    return result


def make_origins(start, end, cadence_hours, now):
    current = datetime.combine(start, time(), tzinfo=UTC)
    stop = min(datetime.combine(end + timedelta(days=1), time(), tzinfo=UTC), now)
    while current < stop:
        yield current
        current += timedelta(hours=cadence_hours)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--end", type=date.fromisoformat, required=True, help="Inclusive UTC end date")
    parser.add_argument("--wavelength", type=int, choices=[94, 131, 171, 193, 211, 304, 335, 1600, 1700, 4500], default=193)
    parser.add_argument("--cadence-hours", type=int, default=6)
    parser.add_argument("--latency-minutes", type=float, default=60, help="Assumed availability delay; not measured (default: 60)")
    parser.add_argument("--max-age-minutes", type=float, default=60, help="Maximum lookback before the latency cutoff")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=45)
    parser.add_argument("--attempts", type=int, default=5)
    parser.add_argument("--ca-file", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("data/aia_synoptic"))
    parser.add_argument("--base-url", default=BASE_URL)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Show the plan without network access or writes")
    args = parser.parse_args()
    now = datetime.now(UTC)
    if not date(2010, 1, 1) <= args.start <= args.end <= now.date():
        parser.error("Use 2010-01-01 <= start <= end <= today (UTC); early mission coverage varies")
    if (args.cadence_hours < 1 or args.workers < 1 or args.attempts < 1 or
            not math.isfinite(args.timeout) or args.timeout <= 0 or
            not math.isfinite(args.latency_minutes) or args.latency_minutes < 0 or
            not math.isfinite(args.max_age_minutes) or not 0 < args.max_age_minutes <= 1440):
        parser.error("Use positive cadence/workers/attempts/timeout, latency >= 0, and 0 < max-age <= 1440")
    if urlparse(args.base_url).scheme != "https":
        parser.error("The archive base URL must use HTTPS")
    origins = list(make_origins(args.start, args.end, args.cadence_hours, now))
    print(f"Planned: {len(origins):,} images at {args.wavelength} Å; about {len(origins) * 1.25 / 1000:.2f} GB.")
    print("Size estimate uses a 193 Å sample; actual size and rejected candidate downloads vary.")
    if args.dry_run:
        return 0
    try:
        import astropy.io.fits  # noqa: F401
        ca_data = read_ca_file(args.ca_file) if args.ca_file else None
    except ImportError:
        parser.error("Install image dependencies: uv sync --locked --group images")
    except (OSError, ValueError) as exc:
        parser.error(f"Invalid CA file: {exc}")
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download_slot, origin, args, ca_data): origin for origin in origins}
        for future in as_completed(futures):
            origin = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {"origin_utc": utc_string(origin), "status": "failed", "error": str(exc)}
            results.append(result)
            detail = result.get("error")
            if result["status"] != "ok" and not detail:
                detail = (result.get("errors") or [{"error": "No acceptable image in the requested time window"}])[-1]["error"]
            print(f"{utc_string(origin)}: {result['status']}" +
                  (" (cache)" if result.get("cached") else "") +
                  (": " + " ".join(detail.split()) if detail else ""), file=sys.stderr)
    results.sort(key=lambda r: r["origin_utc"])
    successful = sum(r["status"] == "ok" for r in results)
    report = {"product": "SDO/AIA synoptic FITS", "base_url": args.base_url,
              "start": args.start.isoformat(), "end_inclusive": args.end.isoformat(),
              "generated_at_utc": utc_string(datetime.now(UTC)), "wavelength_angstrom": args.wavelength,
              "cadence_hours": args.cadence_hours, "assumed_latency_minutes": args.latency_minutes,
              "max_age_before_cutoff_minutes": args.max_age_minutes,
              "requested": len(origins), "successful": successful, "missing_or_failed": len(origins) - successful,
              "note": "Latency is an assumption. Archive observations do not prove historical availability or NRT equivalence.",
              "images": results}
    run_name = f"{args.start}_{args.end}_{args.wavelength:04d}_{args.cadence_hours}h"
    report_path = args.output_dir / "reports" / (run_name + ".json")
    atomic_write(report_path, json.dumps(report, indent=2).encode())
    print(f"{successful}/{len(origins)} origins covered. Report: {report_path}")
    return 0 if successful == len(origins) and successful else 1


if __name__ == "__main__":
    raise SystemExit(main())
