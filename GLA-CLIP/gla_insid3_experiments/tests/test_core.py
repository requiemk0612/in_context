from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from gla_insid3.aligner import AlignerConfig, FACTORIAL, align_feature_windows, factorial_config
from gla_insid3.metrics import binary_metrics, overlap_metrics, seam_metrics
from gla_insid3.pipeline import canonicalize_binary_observations, canonicalize_tensor_observations
from gla_insid3.windows import coverage_map, make_windows, stitch_scores, token_centers
from summarize_results import summarize


class WindowTests(unittest.TestCase):
    def test_edge_anchoring_and_coverage(self) -> None:
        windows = make_windows(896, 896, 512, 256)
        self.assertEqual(len(windows), 9)
        self.assertEqual((windows[-1].x1, windows[-1].y1), (384, 384))
        coverage = coverage_map(windows, 896, 896)
        self.assertEqual(int(coverage.min()), 1)
        self.assertGreaterEqual(int(coverage.max()), 4)

    def test_stride_holes_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            make_windows(512, 512, 128, 129)

    def test_continuous_score_stitch(self) -> None:
        windows = make_windows(4, 6, 4, 2)
        scores = [torch.full((2, 2), float(index + 1)) for index in range(len(windows))]
        stitched, coverage, variance = stitch_scores(scores, windows, (4, 6), "uniform")
        self.assertEqual(tuple(stitched.shape), (4, 6))
        self.assertTrue(torch.all(coverage >= 1))
        self.assertTrue(torch.all(variance >= 0))
        self.assertAlmostEqual(float(stitched[1, 2]), 1.5)


class AlignerTests(unittest.TestCase):
    def test_all_factorial_variants(self) -> None:
        torch.manual_seed(0)
        windows = make_windows(6, 10, 6, 4)
        features = [F.normalize(torch.randn(8, 3, 5), dim=0) for _ in windows]
        coordinates = [token_centers(window, 3, 5) for window in windows]
        for method in FACTORIAL:
            with self.subTest(method=method):
                config = factorial_config(
                    method,
                    AlignerConfig(query_chunk=4, topk=5, coordinate_quantum=1.0),
                )
                outputs, diagnostics = align_feature_windows(features, coordinates, config)
                self.assertEqual([item.shape for item in outputs], [item.shape for item in features])
                self.assertTrue(all(torch.isfinite(item).all() for item in outputs))
                self.assertEqual(len(diagnostics), len(features))
                self.assertIn("outer_attention_mass", diagnostics[0])


class ReplayAndMetricTests(unittest.TestCase):
    def test_canonicalization(self) -> None:
        coords = [torch.tensor([[0.0, 0.0], [1.0, 0.0]])] * 2
        values = [torch.tensor([[1.0, 3.0]]), torch.tensor([[3.0, 5.0]])]
        canonical = canonicalize_tensor_observations(values, coords)
        self.assertTrue(torch.equal(canonical[0], torch.tensor([[2.0, 4.0]])))
        masks = canonicalize_binary_observations(
            [torch.tensor([[False, True]]), torch.tensor([[True, True]])], coords
        )
        self.assertTrue(torch.equal(masks[0], masks[1]))

    def test_overlap_identity(self) -> None:
        feature = F.normalize(torch.arange(1, 13, dtype=torch.float32).reshape(3, 2, 2), dim=0)
        score = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
        candidate = torch.tensor([[False, True], [True, False]])
        forward = score > 0
        nn = torch.tensor([[[0, 1], [2, 3]]])
        membership = torch.tensor([[[False, True], [True, False]]])
        clusters = torch.tensor([[0, 0], [1, 1]])
        coords = torch.tensor([[0.5, 0.5], [1.5, 0.5], [0.5, 1.5], [1.5, 1.5]])
        metrics = overlap_metrics(
            [feature, feature], [score, score], [score, score],
            [candidate, candidate], [forward, forward], [membership, membership],
            [nn, nn], [clusters, clusters], [coords, coords], decision_threshold=0.2,
        )
        self.assertAlmostEqual(metrics["ofc"], 1.0, places=6)
        self.assertEqual(metrics["cwsd"], 0.0)
        self.assertEqual(metrics["cwod_binary"], 0.0)
        self.assertEqual(metrics["candidate_flip_rate"], 0.0)

    def test_ignore_pixels(self) -> None:
        prediction = torch.tensor([[True, False], [False, False]])
        target = torch.zeros((2, 2), dtype=torch.bool)
        ignore = torch.tensor([[True, False], [False, False]])
        result = binary_metrics(prediction, target, ignore)
        self.assertEqual(result["fg_iou"], 0.0)
        self.assertAlmostEqual(result["bg_iou"], 1.0)
        windows = make_windows(2, 2, 1, 1)
        seam = seam_metrics(prediction, target, windows, ignore=ignore)
        self.assertIn("seam_excess_error", seam)

    def test_paired_bootstrap_summary(self) -> None:
        records = [
            {"episode_id": "e1", "method": "B1", "fold": 0, "class_id": 0, "fg_iou": 0.4},
            {"episode_id": "e2", "method": "B1", "fold": 0, "class_id": 0, "fg_iou": 0.5},
            {"episode_id": "e1", "method": "A7", "fold": 0, "class_id": 0, "fg_iou": 0.6},
            {"episode_id": "e2", "method": "A7", "fold": 0, "class_id": 0, "fg_iou": 0.7},
        ]
        result = summarize(records, "B1", samples=100, seed=0)
        delta = result["paired_delta_method_minus_baseline"]["A7"]["fg_iou"]
        self.assertAlmostEqual(delta["mean"], 0.2)
        self.assertEqual(delta["n"], 2)


if __name__ == "__main__":
    unittest.main()
