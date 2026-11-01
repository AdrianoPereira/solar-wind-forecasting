from datetime import date, datetime, timezone, timedelta
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from email.message import Message

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import download_aia_images as aia


class ImageTests(unittest.TestCase):
    def test_index_selection_and_unsafe_links(self):
        origin = datetime(2024, 3, 1, tzinfo=timezone.utc)
        html = ''.join(f'<a href="{name}">file</a>' for name in [
            'AIA20240301_0000_0193.fits', 'AIA20240301_0002_0193.fits',
            'AIA20240301_0000_0211.fits', '../AIA20240301_0000_0193.fits'])
        selected = aia.parse_index(html, 193, origin - timedelta(hours=1), origin)
        self.assertEqual([name for _, name in selected], ['AIA20240301_0000_0193.fits'])

    def test_inclusive_dates_and_cadence(self):
        stamps = list(aia.make_origins(date(2024, 2, 29), date(2024, 3, 1), 6,
                                      datetime(2024, 3, 2, tzinfo=timezone.utc)))
        self.assertEqual(len(stamps), 8)
        self.assertEqual(stamps[-1].hour, 18)

    def payload(self, quality=0, wavelength=193):
        import numpy as np
        from astropy.io import fits
        hdu = fits.PrimaryHDU(np.ones((1024, 1024), dtype='int16'))
        for k, v in {'INSTRUME':'AIA_2', 'DATE-OBS':'2024-03-01T00:00:04.846',
                     'WAVELNTH':wavelength, 'EXPTIME':2.0, 'QUALITY':quality, 'LVL_NUM':1.5}.items():
            hdu.header[k] = v
        out = BytesIO()
        hdu.writeto(out)
        return out.getvalue()

    def test_fits_metadata_and_quality(self):
        stamp, metadata = aia.inspect_fits(self.payload(), 193)
        self.assertEqual(stamp.second, 4)
        self.assertEqual(metadata['data_level'], 1.5)
        for payload in [self.payload(quality=1), self.payload(wavelength=211), b'<html>Error</html>']:
            with self.assertRaises((ValueError, OSError)):
                aia.inspect_fits(payload, 193)

    def test_cache_checksum_and_refetch(self):
        args = SimpleNamespace(refresh=False, wavelength=193, timeout=5, attempts=1)
        with tempfile.TemporaryDirectory() as directory, patch.object(aia, 'fetch', return_value=self.payload()) as fetch:
            path = Path(directory) / 'image.fits'
            self.assertFalse(aia.load_image('https://example.org/a.fits', path, args, None)[2])
            self.assertTrue(aia.load_image('https://example.org/a.fits', path, args, None)[2])
            path.write_bytes(b'corrupted')
            self.assertFalse(aia.load_image('https://example.org/a.fits', path, args, None)[2])
            self.assertEqual(fetch.call_count, 2)

    def test_auxiliary_nan_is_recorded_without_changing_bytes(self):
        from astropy.io import fits
        original = self.payload()
        with fits.open(BytesIO(original)) as hdus:
            hdus[0].header['OSCNMEAN'] = 0.0
            out = BytesIO()
            hdus.writeto(out)
        raw = out.getvalue()
        offset = raw.index(b'OSCNMEAN')
        raw = raw[:offset] + b'OSCNMEAN=                 -nan / Missing auxiliary value'.ljust(80) + raw[offset + 80:]
        before = raw
        _, metadata = aia.inspect_fits(raw, 193)
        self.assertEqual(metadata['header_normalizations'][0]['keyword'], 'OSCNMEAN')
        self.assertEqual(raw, before)
        # The exception must not extend to critical scientific fields.
        offset = raw.index(b'QUALITY ')
        bad = raw[:offset] + b'QUALITY =                  nan'.ljust(80) + raw[offset + 80:]
        with self.assertRaises(Exception):
            aia.inspect_fits(bad, 193)

    def test_temporary_http_error_retries_and_honors_retry_after(self):
        from unittest.mock import MagicMock
        headers = Message()
        headers['Retry-After'] = '12'
        unavailable = HTTPError('https://example.org', 503, 'Unavailable', headers, None)
        response = MagicMock()
        response.__enter__.return_value.read.return_value = b'ok'
        with patch.object(aia, 'urlopen', side_effect=[unavailable, response]) as opened, \
             patch.object(aia.clock_time, 'sleep') as sleep, \
             patch.object(aia.clock_time, 'monotonic', return_value=0), \
             patch.object(aia, 'NEXT_REQUEST', 0):
            self.assertEqual(aia.fetch('https://example.org', 5, 2), b'ok')
            self.assertEqual(opened.call_count, 2)
            sleep.assert_called_with(12)

    def test_header_timestamp_prevents_future_selection(self):
        origin = datetime(2024, 3, 1, tzinfo=timezone.utc)
        args = SimpleNamespace(latency_minutes=0, max_age_minutes=5,
                               base_url='https://example.org/', wavelength=193,
                               timeout=5, attempts=1, output_dir=Path('/tmp'))
        with patch.object(aia, 'fetch', return_value=b'<a href="AIA20240301_0000_0193.fits">x</a>'), \
             patch.object(aia, 'load_image', return_value=(origin + timedelta(seconds=4), {}, False)):
            self.assertEqual(aia.download_slot(origin, args, None)['status'], 'missing')


if __name__ == '__main__':
    unittest.main()
