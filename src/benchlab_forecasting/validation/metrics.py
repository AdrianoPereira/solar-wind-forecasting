"""TIMSE from the participant slide, plus legacy and regression diagnostics."""

import numpy as np

CHALLENGE = {
    "name": "TIMSE", "direction": "minimize", "official_score": None,
    "status": "published_formula_implemented_reference_scorer_pending",
    "source": "https://docs.google.com/document/d/1nLT8KiXn2hRiyoFS9neTDoJuG75ptVTH5VUYa-D8DRw/edit",
    "formula_source": "Participant slide, page 29, IMG_1046.PNG supplied by the user",
    "checked_on": "2026-10-05",
    "faq_sha256": "1183984ad86855f38df6dddc9fe5ab00bc7e74b3158daaa6deab21b1993be325",
    "reason": "Published TIMSE formula implemented locally; platform reference scorer and missing-prediction policy still pending verification.",
}


def _arrays(observed, predicted):
    y, p = np.asarray(observed, dtype=float), np.asarray(predicted, dtype=float)
    if y.ndim != 1 or y.shape != p.shape:
        raise ValueError("Observações e previsões devem ser vetores com o mesmo tamanho")
    return y, p


def regression_metrics(observed, predicted):
    """Pairwise finite values only; report counts to make omissions visible."""
    y, p = _arrays(observed, predicted)
    valid = np.isfinite(y) & np.isfinite(p)
    errors = p[valid] - y[valid]
    mse = float(np.mean(errors**2)) if len(errors) else None
    return {"n": int(valid.sum()), "mse": mse,
            "rmse": float(np.sqrt(mse)) if mse is not None else None,
            "mae": float(np.mean(np.abs(errors))) if len(errors) else None,
            "bias": float(np.mean(errors)) if len(errors) else None}


def timse_profile(observed, predicted):
    """Integrate tail MSE using left rectangles at sorted unique observations.

    Return None for no finite targets or any nonfinite prediction at a finite
    target. This strict missing-data convention is local, not a platform penalty.
    Units are (km/s)^3 for speeds in km/s. No range normalization or lead weights.
    A constant target has zero integration width and score zero by the formula;
    the degenerate flag distinguishes this case from an informative evaluation.
    Sorting and suffix sums cost O(n log n), with O(n) memory.
    """
    y, p = _arrays(observed, predicted)
    finite = np.isfinite(y)
    y, p = y[finite], p[finite]
    if not len(y) or not np.isfinite(p).all():
        return None
    order = np.argsort(y, kind="stable")
    y, p = y[order], p[order]
    thresholds, first = np.unique(y, return_index=True)
    tail_n = len(y) - first
    suffix = np.cumsum(((p - y) ** 2)[::-1])[::-1]
    tail_mse = suffix[first] / tail_n
    widths = np.r_[np.diff(thresholds), 0.0]
    contribution = widths * tail_mse
    return {"timse": float(contribution.sum()), "n": len(y),
            "degenerate": len(thresholds) == 1,
            "threshold": thresholds, "tail_n": tail_n, "tail_mse": tail_mse,
            "width": widths, "contribution": contribution,
            "cumulative_timse": np.cumsum(contribution)}


def timse(observed, predicted):
    """Sum (threshold[j+1] - threshold[j]) * tail_MSE[j], as on the slide."""
    profile = timse_profile(observed, predicted)
    return None if profile is None else profile["timse"]


def timse_weights(observed):
    """Per-pair weights w with TIMSE = sum(w * (prediction - observed)^2).

    w_i = sum over thresholds v_j <= y_i of width_j / tail_n_j, so the weight grows
    with the observed speed. Useful as sample weights to train on the metric.
    Nonfinite observations get weight zero.
    """
    y = np.asarray(observed, dtype=float)
    weights = np.zeros(y.shape)
    finite = np.isfinite(y)
    if not finite.any():
        return weights
    values = y[finite]
    thresholds, counts = np.unique(values, return_counts=True)
    tail_n = len(values) - np.r_[0, np.cumsum(counts)[:-1]]
    cumulative = np.cumsum(np.r_[np.diff(thresholds), 0.0] / tail_n)
    weights[finite] = cumulative[np.searchsorted(thresholds, values)]
    return weights


def timse_proxy(observed, predicted):
    """Local empirical-tail approximation, squared (km/s).

    mean_j mean_{i: y_i >= y_j} (prediction_i - y_i)^2.
    Thresholds are every observed value, including duplicates (empirical mass).
    Missing observations are excluded; a missing prediction at any valid target
    makes the result undefined, preventing selective omission of difficult cases.
    No horizon weights, uptime multiplier, or official equivalence is assumed.
    """
    y, p = _arrays(observed, predicted)
    finite_y = np.isfinite(y)
    y, p = y[finite_y], p[finite_y]
    if not len(y) or not np.isfinite(p).all():
        return None
    order = np.argsort(y, kind="stable")
    y, p = y[order], p[order]
    suffix = np.cumsum(((p - y)**2)[::-1])[::-1]
    first = np.searchsorted(y, y, side="left")
    return float(np.mean(suffix[first] / (len(y) - first)))


def official_timse(*args, **kwargs):
    """Reserved for a verified platform scorer, including its missing-data policy."""
    raise NotImplementedError(CHALLENGE["reason"] + " Use timse for the published formula.")
