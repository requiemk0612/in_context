"""Read-only iSAID access and deterministic episode manifests."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .windows import make_windows


CATEGORIES = [
    "ship", "store_tank", "baseball_diamond", "tennis_court", "basketball_court",
    "Ground_Track_Field", "Bridge", "Large_Vehicle", "Small_Vehicle", "Helicopter",
    "Swimming_pool", "Roundabout", "Soccer_ball_field", "plane", "Harbor",
]


@dataclass(frozen=True)
class Episode:
    episode_id: str
    fold: int
    class_id: int
    class_name: str
    reference_image_ids: list[str]
    target_image_id: str
    window_crop: int
    window_stride: int
    target_windows_with_foreground: int
    target_height: int | None = None
    target_width: int | None = None
    target_foreground_pixels: int | None = None
    target_foreground_fraction: float | None = None
    target_total_windows: int | None = None


class ISAIDStore:
    """Never writes beside the dataset; all metadata remains in experiment outputs."""

    def __init__(self, data_root: str | Path):
        root = Path(data_root).expanduser().resolve()
        self.root = root / "iSAID" if (root / "iSAID").is_dir() else root
        self.image_dir = self.root / "img_dir" / "val"
        self.mask_dir = self.root / "ann_dir" / "val"
        if not self.image_dir.is_dir() or not self.mask_dir.is_dir():
            raise FileNotFoundError(f"Expected iSAID img_dir/val and ann_dir/val under {self.root}")

    def image_path(self, image_id: str) -> Path:
        return self.image_dir / f"{image_id}.png"

    def mask_path(self, image_id: str) -> Path:
        return self.mask_dir / f"{image_id}_instance_color_RGB.png"

    def ids(self) -> list[str]:
        suffix = "_instance_color_RGB.png"
        return sorted(path.name.removesuffix(suffix) for path in self.mask_dir.glob(f"*{suffix}"))

    def load_image(self, image_id: str) -> Image.Image:
        return Image.open(self.image_path(image_id)).convert("RGB")

    def load_label(self, image_id: str) -> np.ndarray:
        with Image.open(self.mask_path(image_id)) as image:
            label = np.asarray(image).copy()
        if label.ndim != 2:
            raise ValueError(
                f"Expected a single-channel semantic label at {self.mask_path(image_id)}, "
                f"got shape {label.shape}"
            )
        return label

    def binary_mask(self, image_id: str, class_id: int) -> torch.Tensor:
        return torch.from_numpy(self.load_label(image_id) == class_id + 1)

    def target_masks(self, image_id: str, class_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the binary class mask and the iSAID void/boundary ignore mask."""
        label = self.load_label(image_id)
        return torch.from_numpy(label == class_id + 1), torch.from_numpy(label == 255)


def scan_class_index(store: ISAIDStore) -> dict[int, list[str]]:
    result = {class_id: [] for class_id in range(len(CATEGORIES))}
    for image_id in store.ids():
        values = np.unique(store.load_label(image_id))
        for value in values:
            class_id = int(value) - 1
            if 0 <= class_id < len(CATEGORIES):
                result[class_id].append(image_id)
    return result


def _foreground_window_count(mask: np.ndarray, crop: int, stride: int) -> int:
    height, width = mask.shape
    return sum(bool(mask[w.y1:w.y2, w.x1:w.x2].any()) for w in make_windows(height, width, crop, stride))


def generate_manifest(
    store: ISAIDStore,
    output_path: str | Path,
    fold: int,
    shots: int,
    num_episodes: int,
    crop: int,
    stride: int,
    seed: int,
    cross_window_only: bool = True,
) -> list[Episode]:
    if fold not in (0, 1, 2):
        raise ValueError("fold must be 0, 1, or 2")
    if shots <= 0 or num_episodes <= 0:
        raise ValueError("shots and num_episodes must be positive")
    # Validate geometry before the potentially expensive label scan.
    make_windows(max(crop, 1), max(crop, 1), crop, stride)
    index = scan_class_index(store)
    rng = random.Random(seed)
    class_ids = list(range(fold * 5, fold * 5 + 5))
    targets: dict[int, list[tuple]] = {}
    for class_id in class_ids:
        items = []
        for image_id in index[class_id]:
            label = store.load_label(image_id)
            mask = label == class_id + 1
            count = _foreground_window_count(mask, crop, stride)
            if not cross_window_only or count >= 2:
                image_height, image_width = mask.shape
                items.append((
                    image_id, count, image_height, image_width,
                    int(mask.sum()), float(mask.mean()),
                    len(make_windows(image_height, image_width, crop, stride)),
                ))
        rng.shuffle(items)
        targets[class_id] = items
    episodes: list[Episode] = []
    cursor = {class_id: 0 for class_id in class_ids}
    while len(episodes) < num_episodes:
        made_progress = False
        for class_id in class_ids:
            if len(episodes) >= num_episodes:
                break
            items = targets[class_id]
            if cursor[class_id] >= len(items):
                continue
            (
                target, window_count, target_height, target_width,
                foreground_pixels, foreground_fraction, total_windows,
            ) = items[cursor[class_id]]
            cursor[class_id] += 1
            refs = [item for item in index[class_id] if item != target]
            if len(refs) < shots:
                continue
            references = rng.sample(refs, shots)
            episodes.append(Episode(
                episode_id=f"f{fold}-c{class_id:02d}-e{len(episodes):04d}",
                fold=fold,
                class_id=class_id,
                class_name=CATEGORIES[class_id],
                reference_image_ids=references,
                target_image_id=target,
                window_crop=crop,
                window_stride=stride,
                target_windows_with_foreground=window_count,
                target_height=target_height,
                target_width=target_width,
                target_foreground_pixels=foreground_pixels,
                target_foreground_fraction=foreground_fraction,
                target_total_windows=total_windows,
            ))
            made_progress = True
        if not made_progress:
            break
    if len(episodes) < num_episodes:
        raise RuntimeError(f"Only {len(episodes)} eligible episodes found; requested {num_episodes}")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for episode in episodes:
            handle.write(json.dumps(asdict(episode), ensure_ascii=False) + "\n")
    return episodes


def load_manifest(path: str | Path) -> list[Episode]:
    with Path(path).open(encoding="utf-8") as handle:
        episodes = [Episode(**json.loads(line)) for line in handle if line.strip()]
    if not episodes:
        raise ValueError(f"Episode manifest is empty: {path}")
    ids = [episode.episode_id for episode in episodes]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Episode manifest contains duplicate episode_id values: {path}")
    for episode in episodes:
        if episode.fold not in (0, 1, 2) or not 0 <= episode.class_id < len(CATEGORIES):
            raise ValueError(f"Invalid fold/class in episode {episode.episode_id}")
    return episodes
