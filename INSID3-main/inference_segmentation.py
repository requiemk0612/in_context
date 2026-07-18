"""In-context segmentation inference script with INSID3."""

import argparse
import datetime
import os
import random
import sys
import time
from os.path import join

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import opts
from datasets import build_dataset
from models import build_insid3_from_args
from utils.metrics import Evaluator, AverageMeter


def main(args: argparse.Namespace) -> float:
    print(args)

    # ──────── Reproducibility and logging setup ────────
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    log_file = join(args.output_dir, 'log.txt')
    with open(log_file, 'w') as fp:
        fp.write(" ".join(sys.argv) + '\n')
        fp.write(str(vars(args)) + '\n\n')

    # ──────── Model setup ────────
    model = build_insid3_from_args(args)
    model.to(args.device)
    model.eval()

    print(f'Parameters: {sum(p.numel() for p in model.parameters()):,}')
    print('Start inference')

    if args.sliding_windows:
        start_time = time.time()
        miou = gla_evaluate(args, model, log_file)
        print(f'Total inference time: {time.time() - start_time:.1f}s')

    else:
        start_time = time.time()
        miou = evaluate(args, model, log_file)
        print(f'Total inference time: {time.time() - start_time:.1f}s')

    return miou

def gla_evaluate(args: argparse.Namespace, model: torch.nn.Module, log_file: str) -> float:
    # Keep dataset construction, metrics, progress reporting, and logging in
    # lockstep with evaluate(); only target feature extraction/prediction is a
    # sliding-window path.
    from utils.DynamicNorm import dynamic_normalization
    from utils.ProxyAnchor import proxy_anchor
    from utils.Sliding import (
        extract_sliding_window_features,
        make_sliding_windows,
        prepare_reference_context,
        segment_feature_windows,
    )
    from utils.k_v_extension import build_global_key_value, key_value_extension

    ds = build_dataset(args.dataset, args=args)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=args.num_workers,
                        collate_fn=lambda x: x[0])
    meter = AverageMeter(args.dataset, ds.class_ids, device=args.device)

    crop_size = int(args.sliding_windows_crop)
    stride = int(args.sliding_windows_stride)
    window_batch_size = int(getattr(args, 'sliding_windows_batch_size', 4))

    proxy_rho = float(args.rho)
    proxy_iterations = int(args.t)
    dn_lambda1 = float(args.lambda1)
    dn_lambda2 = float(args.lambda2)
    if not -1.0 <= proxy_rho <= 1.0:
        raise ValueError('Proxy Anchor rho must be in [-1, 1].')
    if proxy_iterations < 0:
        raise ValueError('Proxy Anchor iterations must be non-negative.')

    use_kve = bool(args.key_value_extension)
    use_proxy = bool(args.proxy_anchor)
    use_dynamic_norm = bool(args.dynamic_normalization)

    pbar = tqdm(loader, ncols=80)
    for idx, batch in enumerate(pbar):

        ref_imgs = batch['ref_imgs']    # list of PIL Images
        ref_masks = batch['ref_masks']  # list of tensors
        tgt_img = batch['tgt_img']      # PIL Image
        tgt_mask = batch['tgt_mask']    # tensor

        try:
            # These calls intentionally match evaluate(). set_target supplies
            # the args.image_size canvas on which crop/stride are measured.
            for i in range(len(ref_imgs)):
                model.set_reference(ref_imgs[i], ref_masks[i])
            model.set_target(tgt_img)
            if model._ref_images is None or model._ref_masks is None or model._tgt_image is None:
                raise RuntimeError('Reference or target state was not initialized.')
            if model._tgt_image.ndim != 4 or model._tgt_image.shape[0] != 1:
                raise ValueError('The transformed target must have shape (1, C, H, W).')

            original_tgt_size = tuple(model._orig_tgt_size)
            target_canvas = model._tgt_image
            canvas_size = tuple(target_canvas.shape[-2:])
            windows = make_sliding_windows(canvas_size, crop_size, stride)

            # Extract reference final patch features once. Target windows are
            # extracted in batches and are kept in row-major window order.
            with torch.no_grad():
                reference = prepare_reference_context(model)
                if reference.debiased_features.shape[1] != len(ref_imgs):
                    raise ValueError('Extracted reference count does not match the episode shots.')
                records = extract_sliding_window_features(
                    model,
                    target_canvas,
                    windows,
                    original_tgt_size,
                    batch_size=window_batch_size,
                    expected_channels=reference.channels,
                )
                window_features = [record.flat_features for record in records]

                if use_kve:
                    key_global, value_global = build_global_key_value(window_features)

                enhanced_features = []
                any_alignment = use_kve or use_proxy or use_dynamic_norm
                for flat_feature in window_features:
                    if not any_alignment:
                        enhanced_features.append(flat_feature)
                        continue

                    if use_kve:
                        query, similarity = key_value_extension(
                            flat_feature, key_global, value_global
                        )
                        keys = key_global
                        values = value_global
                        normalization_window_count = len(window_features)
                    else:
                        # Factor switches remain independent: without KVE,
                        # Proxy/DN attend only within the current window.
                        query = F.normalize(flat_feature, p=2, dim=1)
                        keys = query
                        values = flat_feature
                        similarity = query @ keys.transpose(0, 1)
                        normalization_window_count = 1

                    if use_proxy:
                        _, similarity, positive_count = proxy_anchor(
                            query, keys, proxy_rho, proxy_iterations
                        )
                    elif use_dynamic_norm:
                        positive_count = (similarity > proxy_rho).sum(dim=1)

                    if use_dynamic_norm:
                        enhanced, _ = dynamic_normalization(
                            similarity, values, positive_count,
                            normalization_window_count, dn_lambda1, dn_lambda2,
                        )
                    else:
                        attention = torch.softmax(similarity.float(), dim=1).to(values.dtype)
                        enhanced = attention @ values
                    enhanced_features.append(enhanced)

                pred_mask = segment_feature_windows(
                    model,
                    records,
                    enhanced_features,
                    reference,
                    canvas_size,
                    original_tgt_size,
                )
        finally:
            # evaluate() clears state through model.segment(); this path calls
            # the post-feature stages directly and therefore clears it here.
            model.reset_state()

        tgt_mask = F.interpolate(
            tgt_mask.unsqueeze(0).unsqueeze(0).float(),
            size=pred_mask.shape, mode='nearest',
        ).squeeze(0).squeeze(0) > 0.5

        area_inter, area_union = Evaluator.classify_prediction(
            pred_mask, tgt_mask,
            tgt_ignore_idx=batch.get('tgt_ignore_idx'),
        )
        meter.update(area_inter, area_union, batch['class_id'].to(args.device))

        if (idx + 1) % 50 == 0:
            miou = meter.compute_iou()[0]
            pbar.set_description(f'mIoU: {miou:.1f}')

    miou = meter.compute_iou()[0].item()
    out_str = f'mIoU = {miou:.1f}'
    print(out_str)
    with open(log_file, 'a') as fp:
        fp.write(out_str + '\n')
    return miou

def evaluate(args: argparse.Namespace, model: torch.nn.Module, log_file: str) -> float:
    # ──────── Dataset and loader setup ────────
    ds = build_dataset(args.dataset, args=args)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=args.num_workers,
                        collate_fn=lambda x: x[0])
    meter = AverageMeter(args.dataset, ds.class_ids, device=args.device)

    # ──────── Evaluation loop ────────
    pbar = tqdm(loader, ncols=80)
    for idx, batch in enumerate(pbar):

        ref_imgs = batch['ref_imgs']    # list of PIL Images
        ref_masks = batch['ref_masks']  # list of tensors
        tgt_img = batch['tgt_img']      # PIL Image
        tgt_mask = batch['tgt_mask']    # tensor

        # Set all references
        for i in range(len(ref_imgs)):
            model.set_reference(ref_imgs[i], ref_masks[i])
        # Set target
        model.set_target(tgt_img)
        # Segment
        pred_mask = model.segment()

        tgt_mask = F.interpolate(
            tgt_mask.unsqueeze(0).unsqueeze(0).float(),
            size=pred_mask.shape, mode='nearest',
        ).squeeze(0).squeeze(0) > 0.5

        area_inter, area_union = Evaluator.classify_prediction(
            pred_mask, tgt_mask,
            tgt_ignore_idx=batch.get('tgt_ignore_idx'),
        )
        meter.update(area_inter, area_union, batch['class_id'].to(args.device))

        if (idx + 1) % 50 == 0:
            miou = meter.compute_iou()[0]
            pbar.set_description(f'mIoU: {miou:.1f}')

    # ──────── Final results ────────
    miou = meter.compute_iou()[0].item()
    out_str = f'mIoU = {miou:.1f}'
    print(out_str)
    with open(log_file, 'a') as fp:
        fp.write(out_str + '\n')
    return miou


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'INSID3 inference on in-context segmentation',
        parents=[opts.get_args_parser()],
    )
    args = parser.parse_args()
    timestamp = datetime.datetime.now().strftime('%m%d_%H%M')
    args.output_dir = join(args.output_dir, f'{args.exp_name}_{timestamp}')
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)
