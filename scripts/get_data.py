"""Download and unpack the two public datasets into $VISION_DATA (default ./data).

CUB-200-2011  https://data.caltech.edu/records/65de6-vp158  (about 1.1 GB)
GTSRB         https://benchmark.ini.rub.de/gtsrb_dataset.html, files hosted on sid.erda.dk (about 0.4 GB)

Both datasets keep their own terms of use; nothing from them is redistributed in this repository.
"""
from __future__ import annotations

import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vision.common import DATA  # noqa: E402

FILES = {
    "cub": [("CUB_200_2011.tgz", "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1")],
    "gtsrb": [
        ("GTSRB_Final_Training_Images.zip", "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip"),
        ("GTSRB_Final_Test_Images.zip", "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_Images.zip"),
        ("GTSRB_Final_Test_GT.zip", "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_GT.zip"),
    ],
}


def main() -> None:
    raw = DATA / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    for ds, files in FILES.items():
        dst = DATA / ds
        dst.mkdir(parents=True, exist_ok=True)
        for name, url in files:
            f = raw / name
            if not f.exists():
                print(f"downloading {name}")
                urllib.request.urlretrieve(url, f)
            print(f"unpacking {name}")
            if name.endswith(".tgz"):
                with tarfile.open(f) as t:
                    t.extractall(dst, filter="data")
            else:
                with zipfile.ZipFile(f) as z:
                    z.extractall(dst)
    print(f"done: {DATA}")


if __name__ == "__main__":
    main()
