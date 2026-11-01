# Aquiri Space · Hybrid solar-wind forecast

72 hourly forecasts of the ACE solar-wind bulk speed at L1, issued at `t0`, for
`t0 + 1 h` … `t0 + 72 h`.

Source code, training pipeline and evaluation:
<https://github.com/AdrianoPereira/solar-wind-forecasting>.

## Model

A single LightGBM model covers all 72 lead hours, with the lead hour as a feature. It is
the tuned hybrid of the repository, trained on issuances from Feb 2011 to Dec 2021, with
the number of trees and hyperparameters chosen on 2022–2023. Inputs, all cut off at
`t0 − 1 h`:

- **ACE plasma:** latest speed and its age; 6, 24 and 72 h mean speed; 24 h maximum and
  trend; 24 h mean density and temperature.
- **27-day recurrence:** smoothed speed one and two solar rotations (27 and 54 days)
  before each target hour.
- **Coronal holes:** dark-area fractions (25% and 35% of the disk median) in six 15°
  heliographic-longitude strips, |latitude| < 60°. They come from SDO/AIA 193 Å images
  taken at `t0`, `t0 − 24 h` and `t0 − 48 h`, each the latest image in the hour before its
  own cut-off.

Persistence is blended in with a weight per lead hour, chosen on 2022–2023. The weight is
0.91 at 1 h, 0.40 at 12 h, 0.22 at 24 h and 0.07 at 72 h.

On the Jan 2024 – Oct 2026 test period (4,105 issuances, never used for fitting or
selection):

| Forecast | RMSE (km/s) | RMSE, leads 1–12 h |
| --- | ---: | ---: |
| Submitted (hybrid + persistence blend) | 74.4 | 41.3 |
| Hybrid alone | 75.3 | 48.2 |
| Persistence | 97.8 | — |

## Workflow at each run

1. `run.sh` receives `t0` and calls `forecast.py`.
2. `forecast.py` downloads the ACE hourly SWEPAM files of the last ~60 days from
   `sohoftp.nascom.nasa.gov`. If the hourly file lags, it fills the latest hours from the
   SWPC 1-minute file at `services.swpc.noaa.gov`, using only minutes at least 5 minutes
   before `t0`, and only hours with at least 20 valid minutes.
3. It downloads three 1024 × 1024 AIA 193 Å synoptic FITS files from the JSOC
   near-real-time folder (`jsoc1.stanford.edu`).
   - It falls back to the archive folder when the near-real-time folder lacks the image.
   - The near-real-time quality bit (2³⁰) is ignored.
   - An image with other quality flags is used only when no clean one exists in the window.
4. The feature rows are built by `benchlab_forecasting.hybrid.build_pairs`, the same code
   that built the training data. LightGBM then predicts and the persistence blend is
   applied.
5. The output JSON (`t0`, `valid_from`, 72 values of `forecast_speed_kms`) is uploaded by
   `upload.sh`.

Typical runtime is under one minute, mostly the downloads of about 4 MB.

## Data cut-off

- Only ACE hours up to `t0 − 1 h` and AIA images observed at or before `t − 1 h`
  (for `t = t0, t0 − 24 h, t0 − 48 h`) are used.
- Live data that arrive after `t0` cannot be fetched by a run at `t0`.
- The 1-minute fill uses only records stamped at least 5 minutes before `t0`.

## Failure handling

The run always uploads 72 values:

- **No usable AIA image:** the model trained without coronal-hole features is used.
- **Missing ACE values:** the model takes missing inputs natively. Forecasts that are
  still missing fall back to 27-day recurrence, then persistence.
- **Total failure:** a constant 420 km/s is uploaded.
- **Limits:** forecasts are clipped to 150–2000 km/s, and network access stops after
  10 minutes.

## Contents

| Path | Purpose |
| --- | --- |
| `forecast.py` | live data retrieval, features and prediction |
| `benchlab_forecasting/` | `hybrid.py` (features) and `coronal_holes.py` (image features) from the repository |
| `model/tuned.txt`, `model/tuned.json` | LightGBM booster, feature order and blend weights (with coronal holes) |
| `model/tuned_no_ch.*` | the same without coronal-hole features (fallback) |
| `run.sh`, `upload.sh` | Benchlab interface (upload helper unchanged) |

Dependencies: Python 3.14, numpy, pandas, LightGBM and astropy (`requirements.txt`). The
external domains are declared in `submission.json`.
