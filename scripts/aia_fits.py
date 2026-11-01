"""Read legacy AIA headers without modifying the original FITS bytes."""

from contextlib import contextmanager
import re

# Historical archive metadata sometimes serializes missing auxiliary values as
# bare nan/-nan, which is not a legal FITS numeric value. Do not apply this
# exception to quality, exposure, time, wavelength, image structure, or WCS.
OPTIONAL_NAN_KEYS = frozenset({
    "OSCNMEAN", "OSCNRMS", "RSUN_LF", "X0_LF", "Y0_LF",
    "GCIEC_X", "GCIEC_Y", "GCIEC_Z", "HCIEC_X", "HCIEC_Y", "HCIEC_Z",
})


@contextmanager
def open_aia_fits(source):
    from astropy.io import fits

    with fits.open(source, memmap=False) as hdus:
        notes = []
        for index, hdu in enumerate(hdus):
            for position, card in enumerate(list(hdu.header.cards)):
                if card.keyword not in OPTIONAL_NAN_KEYS:
                    continue
                # Inspect raw card text: accessing .value would fail on bare nan.
                # Astropy's public .image property may auto-fix malformed cards
                # before returning them. Use the retained original card bytes.
                raw = card._image if card._image is not None else card.image
                if re.match(r"^[A-Z0-9_]+\s*=\s*[+-]?nan\s*(?:/|$)", raw, re.IGNORECASE):
                    notes.append({"hdu": index, "keyword": card.keyword,
                                  "original_card": raw,
                                  "action": "Treat auxiliary nonstandard NaN as undefined in memory"})
                    del hdu.header[position]
                    hdu.header.insert(position, (card.keyword, None,
                                                "Missing auxiliary value in original archive header"))
        hdus.verify("exception")
        yield hdus, notes
