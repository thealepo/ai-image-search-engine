from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image

from image_search import discover_images

DATASET_ROOT = Path(__file__).parents[1] / "image_database"


def test_dataset_files_match_metadata_and_are_decodable() -> None:
    image_paths = discover_images(DATASET_ROOT / "images")
    with (DATASET_ROOT / "metadata.csv").open(newline="" , encoding="utf-8-sig") as file:
        metadata_names = {row["filename"] for row in csv.DictReader(file)}

    assert len(image_paths) == 398
    assert {path.name for path in image_paths} == metadata_names

    for path in image_paths:
        with Image.open(path) as image:
            image.verify()
