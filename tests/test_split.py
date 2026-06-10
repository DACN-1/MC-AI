"""Tests for the trajectory-level train/val/test split."""

import unittest

from imitation_learning import split_indices_by_stem


def _samples(stem_sizes: dict[str, int]) -> list[tuple[str, int]]:
    out = []
    for stem, n in stem_sizes.items():
        out.extend((stem, i) for i in range(n))
    return out


class TestSplitByStem(unittest.TestCase):
    def test_partition_is_exact_and_disjoint(self):
        samples = _samples({f"traj_{i}": 50 + i for i in range(40)})
        train, val, test = split_indices_by_stem(samples, 0.1, 0.1)
        all_idx = sorted(train + val + test)
        self.assertEqual(all_idx, list(range(len(samples))))

    def test_no_stem_spans_subsets(self):
        samples = _samples({f"traj_{i}": 30 for i in range(30)})
        train, val, test = split_indices_by_stem(samples, 0.15, 0.15)
        stems = lambda idx: {samples[i][0] for i in idx}  # noqa: E731
        self.assertFalse(stems(train) & stems(val))
        self.assertFalse(stems(train) & stems(test))
        self.assertFalse(stems(val) & stems(test))

    def test_fractions_approximately_honored(self):
        # Many small stems -> realized fractions should track requested ones.
        samples = _samples({f"traj_{i}": 20 for i in range(200)})
        train, val, test = split_indices_by_stem(samples, 0.1, 0.1)
        n = len(samples)
        self.assertAlmostEqual(len(val) / n, 0.1, delta=0.02)
        self.assertAlmostEqual(len(test) / n, 0.1, delta=0.02)
        self.assertGreater(len(train) / n, 0.75)

    def test_deterministic(self):
        samples = _samples({f"traj_{i}": 25 for i in range(50)})
        a = split_indices_by_stem(samples, 0.1, 0.1, seed=42)
        b = split_indices_by_stem(samples, 0.1, 0.1, seed=42)
        self.assertEqual(a, b)
        c = split_indices_by_stem(samples, 0.1, 0.1, seed=7)
        self.assertNotEqual(a, c)

    def test_zero_splits_put_everything_in_train(self):
        samples = _samples({"a": 10, "b": 10})
        train, val, test = split_indices_by_stem(samples, 0.0, 0.0)
        self.assertEqual(len(train), 20)
        self.assertEqual(val, [])
        self.assertEqual(test, [])


if __name__ == "__main__":
    unittest.main()
