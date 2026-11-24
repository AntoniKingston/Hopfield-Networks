import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

DATASETS = [
    "small-7x7",
    "letters-abc-8x12",
    "animals-14x9",
    "OCRA-12x30-cut",
    "large-25x25",
    "large-25x50",
    "large-25x25.plus",
    "letters-14x20",
]


@dataclass
class Dataset:
    name: str
    patterns: np.ndarray
    shape: tuple[int, int]

    @property
    def n_patterns(self) -> int:
        return self.patterns.shape[0]

    @property
    def n_neurons(self) -> int:
        return self.patterns.shape[1]


def shape_from_name(name: str) -> tuple[int, int]:
    """`letters-14x20` -> (20, 14): file names give width x height, arrays use (rows, cols)."""
    match = re.search(r"(\d+)x(\d+)", name)
    if match is None:
        raise ValueError(f"no WxH in dataset name {name!r}")
    width, height = map(int, match.groups())
    return height, width


def load_dataset(name: str, data_dir: Path = DATA_DIR) -> Dataset:
    path = Path(name) if name.endswith(".csv") else data_dir / f"{name}.csv"
    patterns = np.atleast_2d(np.loadtxt(path, delimiter=","))
    shape = shape_from_name(path.stem)
    if patterns.shape[1] != shape[0] * shape[1]:
        raise ValueError(f"{path.name}: {patterns.shape[1]} values do not match shape {shape}")
    return Dataset(path.stem, patterns, shape)


# ---------- large bitmaps ----------


def floyd_steinberg(gray: np.ndarray) -> np.ndarray:
    """Error-diffusion dithering of an image in [0, 1] to {-1, 1}."""
    img = gray.astype(np.float64).copy()
    rows, cols = img.shape
    out = np.empty_like(img)
    for r in range(rows):
        for c in range(cols):
            new = 1.0 if img[r, c] >= 0.5 else 0.0
            err = img[r, c] - new
            out[r, c] = new
            if c + 1 < cols:
                img[r, c + 1] += err * 7 / 16
            if r + 1 < rows:
                if c > 0:
                    img[r + 1, c - 1] += err * 3 / 16
                img[r + 1, c] += err * 5 / 16
                if c + 1 < cols:
                    img[r + 1, c + 1] += err * 1 / 16
    return 2 * out - 1


def binarize(gray: np.ndarray, method: Literal["median", "otsu", "dither"] = "median"):
    """Map a grayscale image in [0, 1] to {-1, 1}.

    `median` splits pixels half/half, so the pattern is balanced (mean ~ 0), which keeps the
    overlap between unrelated images small.
    """
    if method == "median":
        return np.where(gray > np.median(gray), 1.0, -1.0)
    if method == "otsu":
        from skimage.filters import threshold_otsu

        return np.where(gray > threshold_otsu(gray), 1.0, -1.0)
    if method == "dither":
        return floyd_steinberg(gray)
    raise ValueError(f"unknown method: {method}")


def load_images(
    paths: list[str | Path],
    shape: tuple[int, int] = (200, 300),
    method: Literal["median", "otsu", "dither"] = "median",
) -> Dataset:
    """Load photos, crop to the target aspect ratio, resize to `shape` and binarize."""
    from skimage.color import rgb2gray, rgba2rgb
    from skimage.io import imread
    from skimage.transform import resize

    patterns = []
    for path in paths:
        img = imread(path)
        if img.ndim == 3:
            if img.shape[2] == 4:
                img = rgba2rgb(img)
            img = rgb2gray(img)
        img = center_crop(img.astype(np.float64), shape[0] / shape[1])
        if img.max() > 1:
            img = img / 255
        gray = resize(img, shape, anti_aliasing=True)
        patterns.append(binarize(gray, method).ravel())
    return Dataset(f"images-{shape[1]}x{shape[0]}-{method}", np.array(patterns), shape)


def center_crop(img: np.ndarray, aspect: float) -> np.ndarray:
    """Largest centered crop with rows / cols == aspect."""
    rows, cols = img.shape
    if rows / cols > aspect:
        new_rows = round(cols * aspect)
        top = (rows - new_rows) // 2
        return img[top : top + new_rows]
    new_cols = round(rows / aspect)
    left = (cols - new_cols) // 2
    return img[:, left : left + new_cols]


def downsample(dataset: Dataset, factor: int) -> Dataset:
    """Block-average a +-1 dataset by `factor` and re-binarize (ties go to +1)."""
    rows, cols = dataset.shape
    r, c = rows // factor, cols // factor
    X = dataset.patterns.reshape(-1, rows, cols)[:, : r * factor, : c * factor]
    blocks = X.reshape(-1, r, factor, c, factor).mean(axis=(2, 4))
    patterns = np.where(blocks >= 0, 1.0, -1.0).reshape(-1, r * c)
    return Dataset(f"{dataset.name}-down{factor}", patterns, (r, c))


# ---------- statistics ----------


def overlap_matrix(X: np.ndarray) -> np.ndarray:
    """m_{mu nu} = x_mu . x_nu / N."""
    return X @ X.T / X.shape[1]


def hebb_crosstalk(X: np.ndarray) -> np.ndarray:
    """Crosstalk C_i^mu = -x_i^mu sum_{nu != mu} x_i^nu m_{nu mu} (Hertz, Krogh, Palmer).

    With the zero-diagonal Hebb rule, bit i of pattern mu is unstable when C_i^mu > 1 - P/N.
    """
    M = overlap_matrix(X)
    np.fill_diagonal(M, 0)
    return -X * (M @ X)


def pattern_stats(X: np.ndarray) -> dict[str, float]:
    X = np.asarray(X, dtype=np.float64)
    P, N = X.shape
    M = overlap_matrix(X)
    off = np.abs(M[~np.eye(P, dtype=bool)]) if P > 1 else np.array([0.0])
    C = hebb_crosstalk(X)
    eig = np.linalg.eigvalsh(X @ X.T)
    return {
        "P": P,
        "N": N,
        "load": P / N,
        "capacity_0138N": 0.138 * N,
        "bias": float(X.mean()),
        "mean_abs_overlap": float(off.mean()),
        "max_abs_overlap": float(off.max()),
        "rank": int(np.linalg.matrix_rank(X)),
        # condition number of X X^T: sets the number of epochs of Oja's subspace rule
        "cond": float(eig[-1] / eig[0]) if eig[0] > 0 else float("inf"),
        "hebb_unstable_bits": float((C > 1 - P / N).mean()),
        "hebb_unstable_patterns": int((C > 1 - P / N).any(axis=1).sum()),
    }
