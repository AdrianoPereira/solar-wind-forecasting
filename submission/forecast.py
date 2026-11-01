#!/usr/bin/env python3
"""Live 72-hour solar-wind forecast for one issuance time t0 (Benchlab submission).

    python3 forecast.py 20261111T060000Z > forecast.json

1. ACE hourly plasma (NOAA SWPC real-time product) for the last ~60 days, from the monthly
   files on sohoftp.nascom.nasa.gov; the most recent hours are completed from SWPC's 1-minute
   file when the hourly file lags.
2. SDO/AIA 193 Å synoptic images (JSOC near-real-time folder) at t0, t0 − 24 h and t0 − 48 h,
   each the latest image in the hour before the cut-off t − 1 h; coronal-hole fractions as in
   training.
3. The training code builds the 72 feature rows (benchlab_forecasting.hybrid.build_pairs), the
   tuned LightGBM predicts, and persistence is blended in with per-lead weights chosen on
   2022–2023. Without any usable image, the model trained without coronal holes is used.

Nothing observed after the cut-off t0 − 1 h is used. Every failure degrades to a simpler
forecast; the program always prints 72 values and exits 0 unless t0 is malformed.
"""

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from html.parser import HTMLParser
from io import BytesIO
import json
import math
from pathlib import Path
import re
import sys
import time
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from benchlab_forecasting.coronal_holes import aia_geometry, coronal_hole_features  # noqa: E402
from benchlab_forecasting.hybrid import HybridConfig, build_pairs, coronal_hole_columns  # noqa: E402

UTC = timezone.utc
HORIZON = 72
LATENCY = timedelta(hours=1)
HISTORY_DAYS = 60  # recurrence needs 54 days before the target hours, plus rolling windows
ACE_MONTHLY = "https://sohoftp.nascom.nasa.gov/sdb/goes/ace/monthly/{:%Y%m}_ace_swepam_1h.txt"
ACE_MINUTE = "https://services.swpc.noaa.gov/text/ace-swepam.txt"
AIA_BASES = ("https://jsoc1.stanford.edu/data/aia/synoptic/nrt/",
             "https://jsoc1.stanford.edu/data/aia/synoptic/")
AIA_NAME = re.compile(r"AIA(\d{8})_(\d{4})(\d{2})?_0193\.fits")
NRT_QUALITY_BIT = 1 << 30  # set on every near-real-time image; not a defect of the image
MAX_IMAGE_TRIES = 4
FALLBACK_SPEED = 420.0
DEADLINE = time.monotonic() + 600  # leave room in the 15-minute budget for start-up and upload


def log(message):
    print(f"[forecast] {message}", file=sys.stderr, flush=True)


def fetch(url, timeout=30, attempts=3):
    for attempt in range(attempts):
        remaining = DEADLINE - time.monotonic()
        if remaining < 5:
            raise TimeoutError("time budget exhausted")
        try:
            request = Request(url, headers={"User-Agent": "benchlab-hybrid-forecast/1.0"})
            with urlopen(request, timeout=min(timeout, remaining)) as response:
                return response.read()
        except Exception as exc:  # noqa: BLE001 - any network error: retry, then give up
            if getattr(exc, "code", None) == 404 or attempt == attempts - 1:
                raise
            time.sleep(2 * (attempt + 1))


# ---------------------------------------------------------------- ACE plasma

def parse_swepam(text, minute=False):
    """Rows of the SWPC SWEPAM text files with status 0; fill values become NaN."""
    rows = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) != 10 or not parts[0].isdigit() or len(parts[0]) != 4:
            continue
        stamp = datetime(int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3][:2]), int(parts[3][2:]),
                         tzinfo=UTC)
        status = int(parts[6])
        values = [float(v) for v in parts[7:10]]
        if status != 0:
            continue
        density, speed, temperature = (v if v >= 0 else math.nan for v in values)
        rows.append((stamp, speed, density, temperature))
    frame = pd.DataFrame(rows, columns=["timestamp_utc", "speed_km_s", "density_cm3", "temperature_k"])
    return frame.set_index("timestamp_utc")


def load_live_plasma(t0):
    """Hourly speed, density and temperature up to the cut-off hour t0 − 1 h."""
    cutoff = t0 - LATENCY
    months = pd.period_range(pd.Timestamp(t0 - timedelta(days=HISTORY_DAYS)).tz_localize(None),
                             pd.Timestamp(t0).tz_localize(None), freq="M")
    frames = []
    for month in months:
        try:
            frames.append(parse_swepam(fetch(ACE_MONTHLY.format(month.to_timestamp())).decode()))
        except Exception as exc:  # noqa: BLE001
            log(f"ACE monthly {month} unavailable: {exc}")
    hourly = pd.concat(frames) if frames else parse_swepam("")
    hourly = hourly[~hourly.index.duplicated(keep="last")]
    # Complete recent hours from 1-minute data, using only minutes at least 5 min before t0.
    try:
        minute = parse_swepam(fetch(ACE_MINUTE).decode(), minute=True)
        minute = minute[minute.index <= t0 - timedelta(minutes=5)]
        counts = minute.speed_km_s.resample("h").count()
        means = minute.resample("h").mean()[counts >= 20]
        missing = means.index.difference(hourly.dropna(subset=["speed_km_s"]).index)
        if len(missing):
            log(f"hours completed from 1-minute data: {[f'{t:%H}h' for t in missing]}")
            hourly = pd.concat([hourly.drop(missing, errors="ignore"), means.loc[missing]])
    except Exception as exc:  # noqa: BLE001
        log(f"ACE 1-minute file unavailable: {exc}")
    grid = pd.date_range(pd.Timestamp(t0 - timedelta(days=HISTORY_DAYS)).floor("h"), pd.Timestamp(cutoff),
                         freq="h")
    plasma = hourly.sort_index().reindex(grid)
    return plasma


# ---------------------------------------------------------------- AIA coronal holes

class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.names = []

    def handle_starttag(self, tag, attrs):
        href = dict(attrs).get("href") if tag == "a" else None
        if href and AIA_NAME.fullmatch(href):
            self.names.append(href)


@contextmanager
def open_fits(payload):
    from astropy.io import fits

    with fits.open(BytesIO(payload), memmap=False) as hdus:
        hdus.verify("silentfix")
        yield next(h for h in hdus if h.data is not None and h.data.ndim == 2)


def image_features(origin):
    """Coronal-hole fractions from the latest good 193 Å image in [cut-off − 1 h, cut-off]."""
    from astropy.time import Time

    cutoff = origin - LATENCY
    earliest = cutoff - timedelta(hours=1)
    for base in AIA_BASES:
        candidates = []
        for hour in (cutoff.replace(minute=0, second=0, microsecond=0),
                     earliest.replace(minute=0, second=0, microsecond=0)):
            directory = f"{base}{hour:%Y/%m/%d}/H{hour:%H}00/"
            try:
                parser = Links()
                parser.feed(fetch(directory, attempts=2).decode())
            except Exception:  # noqa: BLE001 - folder missing in this tree: try the other
                continue
            for name in parser.names:
                match = AIA_NAME.fullmatch(name)
                stamp = datetime.strptime(match[1] + match[2] + (match[3] or "00"), "%Y%m%d%H%M%S").replace(tzinfo=UTC)
                # A file named at the cut-off minute is observed seconds after it.
                if earliest - timedelta(minutes=1) <= stamp < cutoff:
                    candidates.append((stamp, directory + name))
        flagged = None  # newest image whose only problem is a quality flag
        for _, url in sorted(set(candidates), reverse=True)[:MAX_IMAGE_TRIES]:
            try:
                with open_fits(fetch(url)) as hdu:
                    header = hdu.header
                    if not float(header["EXPTIME"]) > 0:
                        raise ValueError(f"EXPTIME={header['EXPTIME']}")
                    observed = Time(str(header["DATE-OBS"]).rstrip("Z"), format="isot", scale="utc").to_datetime(
                        timezone=UTC)
                    if not earliest <= observed <= cutoff:
                        raise ValueError(f"observed {observed:%H:%M:%S}, outside the window")
                    row = {"origin_utc": pd.Timestamp(origin), "observation_utc": pd.Timestamp(observed),
                           **coronal_hole_features(hdu.data.astype(float), *aia_geometry(header))}
                    quality = int(header["QUALITY"]) & ~NRT_QUALITY_BIT
                if quality == 0:
                    log(f"image for {origin:%Y-%m-%d %H}h: {url.rsplit('/', 1)[1]}")
                    return row
                log(f"flagged {url}: QUALITY={quality}")
                flagged = flagged or (url, row)
            except Exception as exc:  # noqa: BLE001
                log(f"rejected {url}: {exc}")
        # Near-real-time images sometimes carry flags that the archive version of the same
        # observation no longer has; a flagged image is better than none.
        if flagged:
            log(f"image for {origin:%Y-%m-%d %H}h (flagged): {flagged[0].rsplit('/', 1)[1]}")
            return flagged[1]
        if candidates:
            break
    log(f"no usable image for {origin:%Y-%m-%d %H}h")
    return None


# ---------------------------------------------------------------- model

def load_model(name):
    import lightgbm

    meta = json.loads((HERE / "model" / f"{name}.json").read_text())
    return lightgbm.Booster(model_file=str(HERE / "model" / f"{name}.txt")), meta


def forecast(t0, details=None):
    """Blended forecast; `details`, if a dict, receives the inputs and the unblended prediction."""
    plasma = load_live_plasma(t0)
    speed = plasma.speed_km_s.dropna()
    log(f"ACE hours: {speed.size}; latest valid {speed.index.max() if speed.size else None}")
    rows = [r for r in (image_features(t0 - timedelta(hours=lag)) for lag in (0, 24, 48)) if r]
    name = "tuned" if rows else "tuned_no_ch"
    booster, meta = load_model(name)
    config = HybridConfig(use_coronal_holes=bool(rows), latency_hours=1)
    holes = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["origin_utc", "observation_utc", *coronal_hole_columns()])
    frame = build_pairs(plasma, holes, pd.DatetimeIndex([pd.Timestamp(t0)]), config)
    prediction = booster.predict(frame[meta["features"]])
    persistence = frame.last_speed.to_numpy()
    recurrence = frame.recurrence_27d.fillna(frame.last_speed).to_numpy()
    prediction = np.where(np.isfinite(prediction), np.maximum(prediction, 0), recurrence)
    weights = np.asarray(meta["persistence_weight_by_lead"])
    blended = np.where(np.isfinite(persistence), weights * persistence + (1 - weights) * prediction, prediction)
    log(f"model {name}; persistence {persistence[0]:.0f} km/s, age {frame.last_age_h.iloc[0]:.1f} h")
    if details is not None:
        details.update(model=name, frame=frame, holes=holes, prediction=prediction)
    return blended


def main():
    if len(sys.argv) != 2:
        print("usage: forecast.py <YYYYMMDDTHHMMSSZ>", file=sys.stderr)
        return 2
    t0 = datetime.strptime(sys.argv[1], "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
    try:
        speeds = forecast(t0)
    except Exception as exc:  # noqa: BLE001 - never leave the platform without a forecast
        log(f"forecast failed, using {FALLBACK_SPEED} km/s: {type(exc).__name__}: {exc}")
        speeds = np.full(HORIZON, np.nan)
    speeds = np.where(np.isfinite(speeds), np.clip(speeds, 150, 2000), FALLBACK_SPEED)
    json.dump({"t0": f"{t0:%Y-%m-%dT%H:%M:%SZ}", "valid_from": f"{t0 + timedelta(hours=1):%Y-%m-%dT%H:%M:%SZ}",
               "forecast_speed_kms": [round(float(v), 2) for v in speeds]}, sys.stdout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
