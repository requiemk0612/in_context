"""iSAID dataset loader for MMSegmentation-preprocessed 896x896 patches.

Structure:
    /data/lky/data/rs_seg/iSAID/
        img_dir/val/*.png
        ann_dir/val/*_instance_color_RGB.png

Annotation encoding (MMSegmentation iSAID, reduce_zero_label=False):
    0  = background
    1  = ship
    2  = store_tank
    3  = baseball_diamond
    4  = tennis_court
    5  = basketball_court
    6  = Ground_Track_Field
    7  = Bridge
    8  = Large_Vehicle
    9  = Small_Vehicle
    10 = Helicopter
    11 = Swimming_pool
    12 = Roundabout
    13 = Soccer_ball_field
    14 = plane
    15 = Harbor
"""

from __future__ import annotations

import os
import glob
import pickle
import torch
import PIL.Image as Image
import numpy as np
from torch.utils.data import Dataset


class DatasetISAIDNew(Dataset):
    CATEGORIES = [
        "ship", "store_tank", "baseball_diamond", "tennis_court",
        "basketball_court", "Ground_Track_Field", "Bridge",
        "Large_Vehicle", "Small_Vehicle", "Helicopter",
        "Swimming_pool", "Roundabout", "Soccer_ball_field",
        "plane", "Harbor",
    ]

    def __init__(
        self,
        datapath: str,
        shot: int,
        num_test: int = -1,
        fold: int = 0,
    ) -> None:
        self.split = 'val'
        self.benchmark = 'isaid_new'
        self.shot = shot
        self.num_test = num_test
        self.fold = fold
        self.nfolds = 3

        # Keep the dataset root so the classwise metadata cache can be shared by
        # all folds without duplicating it under each image/annotation folder.
        self.datapath = datapath
        self.img_path = os.path.join(datapath, 'img_dir', self.split)
        self.ann_path = os.path.join(datapath, 'ann_dir', self.split)

        self.nclass = len(self.CATEGORIES)
        self.class_ids = self.build_class_ids()

        # Build the complete classwise index first, then expose episodes only
        # for the five classes assigned to the requested fold.
        self.img_metadata_classwise = self.build_img_metadata_classwise()
        self.img_metadata = self.build_img_metadata()

        # Limit evaluation episodes only after applying the fold filter.
        if self.num_test > 0:
            self.img_metadata = self.img_metadata[:self.num_test]

        print(
            f"[DatasetISAIDNew] fold {self.fold}: "
            f"{len(self.img_metadata)} episodes, {len(self.class_ids)} classes"
        )

    def build_class_ids(self) -> list[int]:
        if not 0 <= self.fold < self.nfolds:
            raise ValueError(
                f"iSAID fold must be in [0, {self.nfolds - 1}], got {self.fold}"
            )

        # Follow the iSAID-5i protocol used by datasets/isaid.py: the 15
        # categories are divided into three contiguous folds of five classes.
        nclass_per_fold = self.nclass // self.nfolds
        fold_start = self.fold * nclass_per_fold
        return list(range(fold_start, fold_start + nclass_per_fold))

    def sample_episode(self, idx: int) -> tuple:
        idx %= len(self.img_metadata)
        tgt_name, class_sample = self.img_metadata[idx]

        ref_names = []
        candidates = self.img_metadata_classwise[class_sample]
        while True:
            ref_name = np.random.choice(candidates, 1, replace=False)[0]
            if tgt_name != ref_name:
                ref_names.append(ref_name)
            if len(ref_names) == self.shot:
                break

        return tgt_name, ref_names, class_sample

    def __len__(self) -> int:
        return len(self.img_metadata)

    def read_mask(self, img_name: str) -> torch.Tensor:
        mask_path = os.path.join(self.ann_path, img_name + '_instance_color_RGB.png')
        mask = torch.tensor(np.array(Image.open(mask_path)))
        return mask

    def read_img(self, img_name: str) -> Image.Image:
        return Image.open(os.path.join(self.img_path, img_name + '.png')).convert('RGB')

    def extract_ignore_idx(self, mask: torch.Tensor, class_id: int) -> tuple:
        target_value = class_id + 1

        binary_mask = mask.clone()
        binary_mask[binary_mask != target_value] = 0
        binary_mask[binary_mask == target_value] = 1

        boundary = torch.zeros_like(binary_mask).bool()
        return binary_mask, boundary

    def load_frame(self, tgt_name: str, ref_names: list[str]) -> tuple:
        tgt_img = self.read_img(tgt_name)
        tgt_mask = self.read_mask(tgt_name)
        ref_imgs = [self.read_img(name) for name in ref_names]
        ref_masks = [self.read_mask(name) for name in ref_names]

        org_tgt_imsize = tgt_img.size

        return tgt_img, tgt_mask, ref_imgs, ref_masks, org_tgt_imsize

    def __getitem__(self, idx: int) -> dict:
        idx %= len(self.img_metadata)
        tgt_name, ref_names, class_sample = self.sample_episode(idx)
        tgt_img, tgt_cmask, ref_imgs, ref_cmasks, org_tgt_imsize = self.load_frame(tgt_name, ref_names)

        tgt_mask, tgt_ignore_idx = self.extract_ignore_idx(tgt_cmask.float(), class_sample)

        ref_masks = []
        for scmask in ref_cmasks:
            ref_mask, _ = self.extract_ignore_idx(scmask.float(), class_sample)
            ref_masks.append(ref_mask)

        batch = {
            'tgt_img': tgt_img,
            'tgt_mask': tgt_mask,
            'tgt_ignore_idx': tgt_ignore_idx,
            'ref_imgs': ref_imgs,
            'ref_masks': ref_masks,
            'class_id': torch.tensor(class_sample),
        }

        return batch

    def build_img_metadata(self) -> list:
        items = []
        for class_id in self.class_ids:
            for img_name in self.img_metadata_classwise[class_id]:
                items.append([img_name, class_id])
        return items

    def build_img_metadata_classwise(self) -> dict:
        cache_path = os.path.join(self.datapath, 'classwise_metadata.pkl')
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                metadata = pickle.load(f)
            print(f"[DatasetISAIDNew] Loaded classwise metadata from {cache_path}")
            return metadata

        # Scan all categories, rather than only the active fold, so the same
        # cache remains valid when evaluation is restarted with another fold.
        metadata = {class_id: [] for class_id in range(self.nclass)}
        ann_paths = sorted(glob.glob(os.path.join(self.ann_path, '*_instance_color_RGB.png')))
        for ann_path in ann_paths:
            img_name = os.path.basename(ann_path).replace('_instance_color_RGB.png', '')
            mask = np.array(Image.open(ann_path))
            unique_classes = np.unique(mask)
            for cls in unique_classes:
                if cls == 0:
                    continue
                class_id = int(cls) - 1
                if 0 <= class_id < self.nclass:
                    metadata[class_id].append(img_name)

        # Deduplicate and sort image names to make episode order deterministic.
        metadata = {k: sorted(set(v)) for k, v in metadata.items()}
        with open(cache_path, 'wb') as f:
            pickle.dump(metadata, f)
        print(f"[DatasetISAIDNew] Saved classwise metadata to {cache_path}")
        return metadata


def build(args) -> DatasetISAIDNew:
    # 直接写死用户数据路径
    dataset = DatasetISAIDNew(
        datapath='/data/lky/data/rs_seg/iSAID',
        shot=args.shots,
        num_test=getattr(args, 'num_episodes', -1),
        fold=args.fold,
    )
    return dataset
