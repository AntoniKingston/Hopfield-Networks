"""Download the cat photos used in the large-bitmap experiment into data/cats/.

Sources: one ImageNet sample image per cat class (github.com/EliSchwartz/imagenet-sample-images)
and the `chelsea` image shipped with scikit-image.
"""

import shutil
import urllib.request
from pathlib import Path

import skimage

from hopfield_datasets import DATA_DIR

BASE_URL = "https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/master/"
IMAGENET_CATS = [
    "n02123045_tabby.JPEG",
    "n02123159_tiger_cat.JPEG",
    "n02123394_Persian_cat.JPEG",
    "n02123597_Siamese_cat.JPEG",
    "n02124075_Egyptian_cat.JPEG",
    "n02125311_cougar.JPEG",
    "n02127052_lynx.JPEG",
    "n02128385_leopard.JPEG",
    "n02128757_snow_leopard.JPEG",
]
CATS_DIR = DATA_DIR / "cats"


def main():
    CATS_DIR.mkdir(parents=True, exist_ok=True)
    for name in IMAGENET_CATS:
        target = CATS_DIR / name
        if not target.exists():
            print(f"downloading {name}")
            urllib.request.urlretrieve(BASE_URL + name, target)
    chelsea = Path(skimage.data.__file__).parent / "chelsea.png"
    shutil.copy(chelsea, CATS_DIR / "chelsea.png")
    print(f"{len(list(CATS_DIR.iterdir()))} images in {CATS_DIR}")


if __name__ == "__main__":
    main()
