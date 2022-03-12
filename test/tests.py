import unittest

from src.speech_segmentation import remove_overlapping_segments, equalize_segments


class UtterancesSegmentationTestCase(unittest.TestCase):
    def test_removing_overlap_type1(self):
        """
        source: |-----------|
        target:          |---------|
        """
        source_intervals = [(0, 10)]
        target_intervals = [(7, 15)]

        [source_intervals, target_intervals] = remove_overlapping_segments(source_intervals, target_intervals)
        self.assertEqual(source_intervals, [(0, 7)])
        self.assertEqual(target_intervals, [(7, 15)])

    def test_removing_overlap_type2(self):
        """
        source:         |-----------|
        target:  |---------|
        """
        source_intervals = [(7, 15)]
        target_intervals = [(0, 10)]

        [source_intervals, target_intervals] = remove_overlapping_segments(source_intervals, target_intervals)
        self.assertEqual(source_intervals, [(10, 15)])
        self.assertEqual(target_intervals, [(0, 10)])

    def test_removing_overlap_type3(self):
        """
        source: |-----------|
        target:    |-----|
        """
        source_intervals = [(0, 10)]
        target_intervals = [(3, 6)]

        [source_intervals, target_intervals] = remove_overlapping_segments(source_intervals, target_intervals)
        self.assertEqual(source_intervals, [(0, 3), (6, 10)])
        self.assertEqual(target_intervals, [(3, 6)])

    def test_removing_overlap_type4(self):
        """
        source:    |-----|
        target: |-----------|
        """
        source_intervals = [(3, 6)]
        target_intervals = [(0, 10)]

        [source_intervals, target_intervals] = remove_overlapping_segments(source_intervals, target_intervals)
        self.assertEqual(source_intervals, [])
        self.assertEqual(target_intervals, [(0, 10)])

    def test_removing_overlap_type5(self):
        """
        source:    |-----|
        target: |-----------|
        """
        source_intervals = [(0, 10)]
        target_intervals = [(0, 10)]

        [source_intervals, target_intervals] = remove_overlapping_segments(source_intervals, target_intervals)
        self.assertEqual(source_intervals, [])
        self.assertEqual(target_intervals, [(0, 10)])

    def test_equalization_type1(self):
        """
        source: |-----| |-----| |-----|
        target:                         |----| |----|
        """
        source_intervals = [(0, 5), (8, 10), (12, 16)]
        target_intervals = [(20, 25), (28, 30)]

        [source_intervals, target_intervals] = equalize_segments(source_intervals, target_intervals)
        self.assertEqual(source_intervals, [(12, 16)])
        self.assertEqual(target_intervals, [(20, 25)])

    def test_equalization_type2(self):
        """
        source:                         |----| |----|
        target: |-----| |-----| |-----|
        """
        source_intervals = [(20, 25), (28, 30)]
        target_intervals = [(0, 5), (8, 10), (12, 16)]

        [source_intervals, target_intervals] = equalize_segments(source_intervals, target_intervals)
        self.assertEqual(source_intervals, [])
        self.assertEqual(target_intervals, [])

    def test_equalization_type3(self):
        """
        source:                 |----| |----|
        target: |-----| |-----|               |-----| |-----|
        """
        source_intervals = [(12, 16), (18, 24)]
        target_intervals = [(0, 5), (8, 10), (30, 35), (37, 39)]

        [source_intervals, target_intervals] = equalize_segments(source_intervals, target_intervals)
        self.assertEqual(source_intervals, [(18, 24)])
        self.assertEqual(target_intervals, [(30, 35)])

    def test_equalization_type4(self):
        """
        source: |-----| |-----|               |-----| |-----|
        target:                 |----| |----|
        """
        source_intervals = [(0, 5), (8, 10), (30, 35), (37, 39)]
        target_intervals = [(12, 16), (18, 24)]

        [source_intervals, target_intervals] = equalize_segments(source_intervals, target_intervals)
        self.assertEqual(source_intervals, [(8, 10)])
        self.assertEqual(target_intervals, [(12, 16)])

    def test_segmentation(self):
        """
        source: |-----| |-----|          |------| |--|        |-----|    |-----|   |-----|       |-----|     |-----|
        target:     |-----|    |--| |---|  |--|        |------------------|   |-----|
        """
        source_intervals = [(0, 5), (8, 12), (22, 28), (29, 30), (32, 37), (40, 45), (50, 55), (65, 70), (75, 80)]
        target_intervals = [(3, 9), (14, 16), (18, 20), (24, 26), (31, 42), (47, 52)]

        [source_intervals, target_intervals] = remove_overlapping_segments(source_intervals, target_intervals)
        self.assertEqual(source_intervals,
                         [(0, 3), (9, 12), (22, 24), (26, 28), (29, 30), (42, 45), (52, 55), (65, 70), (75, 80)])
        self.assertEqual(target_intervals, [(3, 9), (14, 16), (18, 20), (24, 26), (31, 42), (47, 52)])

        [source_intervals, target_intervals] = equalize_segments(source_intervals, target_intervals)
        self.assertEqual(source_intervals, [(0, 3), (9, 12), (22, 24), (29, 30), (42, 45)])
        self.assertEqual(target_intervals, [(3, 9), (14, 16), (24, 26), (31, 42), (47, 52)])


if __name__ == "__main__":
    unittest.main()
