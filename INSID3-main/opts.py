"""Command-line arguments for INSID3 inference."""

import argparse

SUPPORTED_DATASETS = [
    "coco", "lvis", "pascal_part", "paco_part",
    "isaid", "isic", "lung", "suim", "permis",
    "isaid_new",
]


def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("INSID3 inference", add_help=False)

    # Model
    parser.add_argument(
        "--model-size",
        default="large",
        choices=["small", "base", "large"],
        help="DINOv3 backbone size",
    )
    parser.add_argument(
        "--image-size",
        default=1024,
        type=int,
        help="Input image resolution",
    )
    parser.add_argument(
        "--crf-mask-refinement", "-crf",
        action="store_true",
        help="Enable CRF-based mask refinement.",
    )
    parser.add_argument(
        "--crf-size",
        default=640,
        type=int,
        help="Spatial resolution used for CRF mask refinement.",
    )

    # Episode
    parser.add_argument(
        "--shots",
        default=1,
        type=int,
        help="Number of reference images (shots)",
    )

    # Hyperparameters
    parser.add_argument(
        "--svd-comps",
        default=500,
        type=int,
        help="Number of SVD components for positional debiasing",
    )
    parser.add_argument(
        "--tau",
        default=0.6,
        type=float,
        help="Clustering distance threshold",
    )
    parser.add_argument(
        "--merge-thresh",
        default=0.2,
        type=float,
        help="Cluster aggregation threshold",
    )

    # Dataset
    parser.add_argument(
        "--dataset",
        default="isaid_new",
        choices=SUPPORTED_DATASETS,
        help="Dataset for evaluation",
    )
    parser.add_argument(
        "--data-root",
        default="/data/lky/data/rs_seg/",
        help="Root directory of datasets",
    )
    parser.add_argument(
        "--fold",
        default=0,
        type=int,
        help="Fold index: for COCO, LVIS, iSAID, PASCAL-Part, PACO-Part",
    )

    # Runtime
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Directory for logs and results",
    )
    parser.add_argument(
        "--exp-name",
        default="insid3-iSAID",
        help="Run name",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device to use (cuda or cpu)",
    )
    parser.add_argument(
        "--seed",
        default=0,
        type=int,
        help="Random seed",
    )
    
    parser.add_argument(
        "--num-episodes",
        default=-1,
        type=int,
        help="Limit number of evaluation episodes (-1 = all)",
    )

    parser.add_argument(
        "--num-workers",
        default=0,
        type=int,
        help="Number of data loading workers",
    )

    # sliding windows
    parser.add_argument(
        "--sliding-windows", "-sw",
        action="store_true",
        help="sliding windows",
    )

    parser.add_argument(
        "--sliding-windows-crop", "-swc",
        default=256,
        type=int,
        help="the size of sliding windows",
    )

    parser.add_argument(
        "--sliding-windows-stride", "-sws",
        default=128,
        type=int,
        help="the strides of sliding windows",
    )
    
    # GLA-CLIP
    parser.add_argument(
        "--key-value-extension", "-kve",
        action="store_true",
        help="whether or not utilize key-value-extension",
    )

    parser.add_argument(
        "--proxy-anchor", "-pa",
        action="store_true",
        help="whether or not utilize proxy-anchor",
    )

    parser.add_argument(
        "--dynamic-normalization", "-dn",
        action="store_true",
        help="whether or not utilize dynamic-normalization",
    )

    # SPAR
    parser.add_argument(
        "--spar",
        action="store_true",
        help="whether or not utilize distillation",
    )

    parser.add_argument(
        "--rho",
        default=0.6,
        type=float,
        help="proxy anchor hyperparameter 1",
    )

    parser.add_argument(
        "--t",
        default=2,
        type=int,
        help="proxy anchor hyperparameter 2",
    )

    parser.add_argument(
        "--lambda1",
        default=0.3,
        type=float,
        help="dynamic normalization hyperparameter 1",
    )

    parser.add_argument(
        "--lambda2",
        default=30.0,
        type=float,
        help="dynamic normalization hyperparameter 2",
    )

    return parser
