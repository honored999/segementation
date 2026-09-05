"""Tests for selecting a CI-1 FLAIR DICOM series."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parents[1]))
from ci1_nvauto_common import DicomSeries, select_flair_series


class FlairSelectionTest(unittest.TestCase):
    def test_prefers_regular_flair_and_excludes_t1_flair(self) -> None:
        flair = DicomSeries(uid="flair", files=[Path("flair.dcm")], series_description="Axi T2 FLAIR", has_pixel_data=True)
        t1_flair = DicomSeries(uid="t1", files=[Path("t1.dcm")] * 10, series_description="T1 FLAIR", has_pixel_data=True)
        localizer = DicomSeries(uid="loc", files=[Path("loc.dcm")] * 20, series_description="FLAIR Localizer", has_pixel_data=True)

        selected = select_flair_series([t1_flair, localizer, flair])

        self.assertIs(selected, flair)


if __name__ == "__main__":
    unittest.main()
