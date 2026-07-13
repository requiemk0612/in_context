from __future__ import annotations

import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from gla_insid3.aligner import AlignerConfig
from gla_insid3.aligner import FACTORIAL
from gla_insid3.pipeline import (
    I1_extract_windows,
    I2_align_features,
    I3_reason_per_window,
    I4_stitch_and_refine,
    prepare_reference,
    run_early_reasoning,
)
from gla_insid3.windows import make_windows
from run_experiment import _method_result, _reason_windows


def fake_external_modules() -> dict[str, types.ModuleType]:
    data = types.ModuleType("utils.data")

    def downsample_mask(mask: torch.Tensor, h: int, w: int) -> torch.Tensor:
        return F.interpolate(mask.float(), (h, w), mode="nearest")[0, 0] > 0.5

    data.downsample_mask = downsample_mask
    clustering = types.ModuleType("utils.clustering")

    def agglomerative_clustering(features: torch.Tensor, tau: float) -> torch.Tensor:
        del tau
        return (features[:, 0] > features[:, 0].median()).long()

    def compute_cluster_prototypes(features: torch.Tensor, labels: torch.Tensor, count: int) -> torch.Tensor:
        prototypes = []
        for label in range(count):
            selected = features[labels == label]
            prototypes.append(F.normalize(selected.mean(dim=0), dim=0))
        return torch.stack(prototypes)

    clustering.agglomerative_clustering = agglomerative_clustering
    clustering.compute_cluster_prototypes = compute_cluster_prototypes
    return {"utils.data": data, "utils.clustering": clustering}


class FakeModel:
    image_size = 16
    tau = 0.6
    merge_threshold = 0.2

    @staticmethod
    def _transform(image: Image.Image) -> torch.Tensor:
        array = np.asarray(image.resize((16, 16)), dtype=np.float32).copy()
        return torch.from_numpy(array).permute(2, 0, 1) / 255.0

    @staticmethod
    def _extract_features(images: torch.Tensor) -> torch.Tensor:
        batch, items = images.shape[:2]
        flat = images.reshape(batch * items, *images.shape[2:])
        pooled = F.adaptive_avg_pool2d(flat, (4, 4))
        features = torch.cat((pooled, pooled.square()), dim=1)
        return features.reshape(batch, items, 6, 4, 4)

    @staticmethod
    def _debias_features(features: torch.Tensor) -> torch.Tensor:
        centered = features - 0.1 * features.mean(dim=2, keepdim=True)
        return F.normalize(centered, dim=2)


class PipelineSmokeTest(unittest.TestCase):
    def test_four_interfaces_without_dino_weights(self) -> None:
        yy, xx = np.mgrid[:24, :24]
        array = np.stack((xx * 10, yy * 10, (xx + yy) * 5), axis=-1).clip(0, 255).astype(np.uint8)
        target = Image.fromarray(array, mode="RGB")
        reference_image = Image.fromarray(np.roll(array, 2, axis=0), mode="RGB")
        reference_mask = torch.zeros((24, 24), dtype=torch.bool)
        reference_mask[6:18, 6:18] = True
        model = FakeModel()
        windows = make_windows(24, 24, 16, 8)

        with patch.dict(sys.modules, fake_external_modules()):
            reference = prepare_reference(model, [reference_image], [reference_mask], "cpu")
            state = I1_extract_windows(model, target, windows, "cpu", batch_size=2)
            aligned, diagnostics = I2_align_features(
                state,
                AlignerConfig(
                    kve=True, proxy=True, dynamic_norm=True,
                    query_chunk=5, coordinate_quantum=1.0,
                ),
            )
            results = [
                I3_reason_per_window(model, reference, raw, semantic)
                for raw, semantic in zip(state.raw, aligned)
            ]
            stitched = I4_stitch_and_refine(model, target, results, windows, "uniform")
            early = run_early_reasoning(model, reference, state, (24, 24), max_tokens=64)

        self.assertEqual(tuple(stitched["stitched_score"].shape), (24, 24))
        self.assertEqual(int(stitched["coverage"].min()), 1)
        self.assertEqual(len(diagnostics), len(windows))
        self.assertEqual(tuple(early["continuous_score"].shape), (6, 6))
        required = {
            "raw_feat", "debiased_feat", "sim_fwd", "nn_ref_index",
            "candidate_mask", "cluster_labels", "seed_id", "combined_score",
            "continuous_score", "pre_crf_mask",
        }
        self.assertTrue(required.issubset(results[0]))

    def test_runner_method_matrix(self) -> None:
        yy, xx = np.mgrid[:24, :24]
        array = np.stack((xx * 10, yy * 10, (xx + yy) * 5), axis=-1).clip(0, 255).astype(np.uint8)
        target = Image.fromarray(array, mode="RGB")
        mask = torch.zeros((24, 24), dtype=torch.bool)
        mask[5:19, 5:19] = True
        model = FakeModel()
        windows = make_windows(24, 24, 16, 8)
        args = SimpleNamespace(
            device="cpu", early_max_tokens=64, d4_max_tokens=64,
            coordinate_quantum=1.0, proxy_rho=0.6, proxy_iters=2,
            dn_lambda1=0.3, dn_lambda2=30.0, fixed_beta=1.2,
            fixed_gamma=3.0, dn_cutoff=0.0, token_bank="duplicate",
            topk=8, query_chunk=5, attention_temperature=1.0,
            enable_crf=False,
        )
        methods = ["B0", "B1", "B2", "B3", *FACTORIAL]
        methods += ["R-D1", "R-D2", "R-D3", "R-D4", "R-D5"]
        with patch.dict(sys.modules, fake_external_modules()):
            reference = prepare_reference(model, [target], [mask], "cpu")
            state = I1_extract_windows(model, target, windows, "cpu", batch_size=2)
            base_results = _reason_windows(model, reference, state.raw, state.debiased)
            for method in methods:
                with self.subTest(method=method):
                    results, _, stitched, _ = _method_result(
                        method, model, reference, target, state, args, base_results
                    )
                    self.assertTrue(results)
                    self.assertEqual(tuple(stitched["post_crf_mask"].shape), (24, 24))


if __name__ == "__main__":
    unittest.main()
