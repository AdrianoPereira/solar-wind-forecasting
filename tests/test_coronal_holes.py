import unittest

import numpy as np

from benchlab_forecasting.coronal_holes import (
    aia_geometry, coronal_hole_features, feature_names, heliographic_grid,
)


class CoronalHoleTests(unittest.TestCase):
    def setUp(self):
        self.shape, self.center, self.radius = (401, 401), 200.0, 180.0

    def test_grid_center_and_limb(self):
        longitude, latitude, mu = heliographic_grid(self.shape, self.center, self.center, self.radius, 0)
        self.assertAlmostEqual(longitude[200, 200], 0)
        self.assertAlmostEqual(latitude[200, 200], 0)
        self.assertAlmostEqual(mu[200, 200], 1)
        # Image rows grow northward and columns westward (north up, as in AIA level 1.5).
        self.assertGreater(latitude[300, 200], 0)
        self.assertGreater(longitude[200, 300], 0)
        self.assertTrue(np.isnan(longitude[0, 0]))
        # Tilting the north pole toward the observer shifts disk center to latitude B0.
        _, tilted, _ = heliographic_grid(self.shape, self.center, self.center, self.radius, 7)
        self.assertAlmostEqual(tilted[200, 200], 7)

    def test_dark_patch_lands_in_its_longitude_bin(self):
        longitude, latitude, _ = heliographic_grid(self.shape, self.center, self.center, self.radius, 0)
        image = np.full(self.shape, 100.0)
        hole = (longitude > -40) & (longitude < -32) & (np.abs(latitude) < 10)
        image[hole] = 10
        features = coronal_hole_features(image, self.center, self.center, self.radius, 0)
        self.assertEqual(set(features) - {"disk_median"}, set(feature_names()))
        self.assertEqual(features["disk_median"], 100)
        # True-area fraction of the bin: (8/15 of its longitude) x (sin 10° / sin 30°).
        expected = 8 / 15 * np.sin(np.radians(10)) / np.sin(np.radians(30))
        self.assertAlmostEqual(features["ch_t35_lat30_lon-45_-30"], expected, delta=0.03 * expected)
        for name, value in features.items():
            if name.startswith("ch_") and "lon-45_-30" not in name:
                self.assertEqual(value, 0, name)
        # The same patch is a smaller fraction of the wider latitude band.
        self.assertLess(features["ch_t35_lat60_lon-45_-30"], features["ch_t35_lat30_lon-45_-30"])

    def test_foreshortening_correction(self):
        """Equal true areas give equal fractions near center and near the limb."""
        longitude, latitude, _ = heliographic_grid((1601, 1601), 800, 800, 780, 0)
        image = np.full(longitude.shape, 100.0)
        for west, east in ((1, 14), (-59, -46)):
            image[(longitude > west) & (longitude < east) & (np.abs(latitude) < 5)] = 10
        features = coronal_hole_features(image, 800, 800, 780, 0)
        center = features["ch_t35_lat30_lon+0_+15"]
        limb = features["ch_t35_lat30_lon-60_-45"]
        self.assertAlmostEqual(center, limb, delta=0.02 * center)
        # Without the 1/mu weight the limb patch would look much smaller in pixels.
        pixels = lambda w, e: ((longitude > w) & (longitude < e) & (np.abs(latitude) < 5)).sum()
        self.assertLess(pixels(-59, -46), 0.7 * pixels(1, 14))

    def test_aia_geometry(self):
        header = {"CROTA2": 0.0, "CDELT1": 2.4, "CDELT2": 2.4, "CRPIX1": 512.5, "CRPIX2": 512.5,
                  "CRVAL1": 0.0, "CRVAL2": 0.0, "RSUN_OBS": 976.06, "CRLT_OBS": -2.97}
        center_x, center_y, radius, b0 = aia_geometry(header)
        self.assertEqual((center_x, center_y), (511.5, 511.5))
        self.assertAlmostEqual(radius, 406.69, places=2)
        self.assertEqual(b0, -2.97)
        with self.assertRaisesRegex(ValueError, "north-up"):
            aia_geometry({**header, "CROTA2": 90.0})


if __name__ == "__main__":
    unittest.main()
