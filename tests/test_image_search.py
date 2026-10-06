from __future__ import annotations

from pathlib import Path

import pytest
import torch
from PIL import Image

from app import create_demo
from image_search import ImageSearchEngine , discover_images , load_image


class FakeClipModel:
    """Small deterministic replacement for CLIP used by the unit tests."""

    def encode(self , value , **kwargs):
        if isinstance(value , str):
            vectors = {
                "red": torch.tensor([1.0 , 0.0]) ,
                "blue": torch.tensor([0.0 , 1.0]) ,
            }
            return vectors.get(value , torch.tensor([1.0 , 1.0])).float()

        vectors = []
        for image in value:
            red , _green , blue = image.resize((1 , 1)).getpixel((0 , 0))
            vector = torch.tensor([float(red) , float(blue)])
            vectors.append(vector / vector.norm())
        return torch.stack(vectors)


def make_image(path: Path , color: str) -> None:
    Image.new("RGB" , (10 , 10) , color=color).save(path)


def test_discover_and_load_images(tmp_path: Path) -> None:
    make_image(tmp_path / "b.png" , "blue")
    make_image(tmp_path / "a.jpg" , "red")
    (tmp_path / "notes.txt").write_text("not an image")

    paths = discover_images(tmp_path)

    assert [path.name for path in paths] == ["a.jpg" , "b.png"]
    assert load_image(paths[0]).mode == "RGB"


def test_missing_directory_is_reported(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError , match="does not exist"):
        discover_images(tmp_path / "missing")


def test_build_index_and_rank_results(tmp_path: Path) -> None:
    make_image(tmp_path / "red.jpg" , "red")
    make_image(tmp_path / "blue.jpg" , "blue")

    engine = ImageSearchEngine.from_directory(
        tmp_path , model=FakeClipModel() , batch_size=1
    )

    results = engine.search("red" , top_k=10)
    assert [result.path.name for result in results] == ["red.jpg" , "blue.jpg"]
    assert results[0].score == pytest.approx(1.0)


def test_unreadable_images_are_skipped(tmp_path: Path) -> None:
    make_image(tmp_path / "valid.jpg" , "red")
    (tmp_path / "broken.jpg").write_bytes(b"not a jpeg")

    with pytest.warns(UserWarning , match="Skipping unreadable image"):
        engine = ImageSearchEngine.from_directory(tmp_path , model=FakeClipModel())

    assert [path.name for path in engine.image_paths] == ["valid.jpg"]


def test_empty_queries_are_rejected(tmp_path: Path) -> None:
    make_image(tmp_path / "red.jpg" , "red")
    engine = ImageSearchEngine.from_directory(tmp_path , model=FakeClipModel())

    with pytest.raises(ValueError , match="Enter a search query"):
        engine.search("  ")


def test_gradio_interface_returns_gallery_items(tmp_path: Path) -> None:
    make_image(tmp_path / "red.jpg" , "red")
    engine = ImageSearchEngine.from_directory(tmp_path , model=FakeClipModel())
    demo = create_demo(engine)

    assert demo.fn("red" , 1) == [
        (str(tmp_path / "red.jpg") , "red.jpg - Similarity: 1.000")
    ]
