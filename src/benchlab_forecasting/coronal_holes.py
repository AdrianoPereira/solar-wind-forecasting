"""Coronal-hole area fractions from a full-disk EUV image (e.g. SDO/AIA 193 Å).

Coronal holes are the dark, open-field regions that feed the fast solar wind.
Wind leaving a hole near the central meridian reaches L1 roughly 3–5 days later,
and the Sun rotates about 13.2° per day, so holes east of the central meridian
announce wind that arrives later still. The features below are area fractions of
dark pixels in heliographic longitude bins, corrected for foreshortening.
"""

import numpy as np


# Longitude bins (degrees, Stonyhurst, negative = east) and latitude bands.
LONGITUDE_EDGES = (-60, -45, -30, -15, 0, 15, 30)
LATITUDE_BANDS = (30, 60)
# Dark-pixel thresholds as fractions of the on-disk median intensity.
THRESHOLDS = (0.25, 0.35, 0.45)


def feature_names():
    names = []
    for threshold in THRESHOLDS:
        for band in LATITUDE_BANDS:
            for west, east in zip(LONGITUDE_EDGES[:-1], LONGITUDE_EDGES[1:]):
                names.append(f"ch_t{int(threshold * 100)}_lat{band}_lon{west:+d}_{east:+d}")
    return names


def heliographic_grid(shape, center_x, center_y, radius_px, b0_deg):
    """Stonyhurst longitude/latitude and mu = cos(heliocentric angle) per pixel.

    Uses the orthographic projection, accurate to well under a pixel at 1 AU.
    Off-disk pixels get NaN coordinates and mu = 0.
    """
    rows, columns = np.indices(shape, dtype=float)
    x = (columns - center_x) / radius_px
    y = (rows - center_y) / radius_px
    rho2 = x * x + y * y
    on_disk = rho2 < 1
    z = np.sqrt(np.where(on_disk, 1 - rho2, 0))
    b0 = np.deg2rad(b0_deg)
    latitude = np.degrees(np.arcsin(np.clip(y * np.cos(b0) + z * np.sin(b0), -1, 1)))
    longitude = np.degrees(np.arctan2(x, z * np.cos(b0) - y * np.sin(b0)))
    latitude[~on_disk] = np.nan
    longitude[~on_disk] = np.nan
    return longitude, latitude, z


def coronal_hole_features(image, center_x, center_y, radius_px, b0_deg, min_mu=0.2):
    """Area-weighted dark fractions per longitude bin, latitude band and threshold.

    `image` must have solar north up; `center_x/center_y` are 0-based pixel
    coordinates of disk center. Pixels with mu below `min_mu` (near the limb) are
    ignored. Returns a dict keyed by `feature_names()` plus the disk median.
    """
    image = np.asarray(image, dtype=float)
    longitude, latitude, mu = heliographic_grid(image.shape, center_x, center_y, radius_px, b0_deg)
    usable = (mu >= min_mu) & np.isfinite(image)
    if usable.sum() < 1000:
        raise ValueError("Too few usable on-disk pixels")
    median = float(np.median(image[usable]))
    if not np.isfinite(median) or median <= 0:
        raise ValueError("On-disk median intensity must be positive")
    # A pixel's projected area grows by 1/mu when mapped back onto the sphere.
    weight = np.where(usable, 1 / np.maximum(mu, min_mu), 0)
    features = {"disk_median": median}
    for threshold in THRESHOLDS:
        dark = image < threshold * median
        for band in LATITUDE_BANDS:
            in_band = usable & (np.abs(latitude) < band)
            for west, east in zip(LONGITUDE_EDGES[:-1], LONGITUDE_EDGES[1:]):
                region = in_band & (longitude >= west) & (longitude < east)
                total = weight[region].sum()
                name = f"ch_t{int(threshold * 100)}_lat{band}_lon{west:+d}_{east:+d}"
                features[name] = float(weight[region & dark].sum() / total) if total > 0 else np.nan
    return features


def aia_geometry(header):
    """Disk center (0-based pixels), radius in pixels and B0 from an AIA level-1.5 header."""
    if abs(float(header.get("CROTA2") or 0)) > 0.5:
        raise ValueError("Image is not north-up; rotate it before extracting features")
    scale_x, scale_y = float(header["CDELT1"]), float(header["CDELT2"])
    center_x = header["CRPIX1"] - 1 - float(header.get("CRVAL1") or 0) / scale_x
    center_y = header["CRPIX2"] - 1 - float(header.get("CRVAL2") or 0) / scale_y
    radius_px = float(header["RSUN_OBS"]) / scale_x
    return center_x, center_y, radius_px, float(header["CRLT_OBS"])
