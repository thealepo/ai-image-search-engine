"""Core image indexing and semantic search functionality."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any , Iterable
import warnings

import torch
from PIL import Image , ImageOps , UnidentifiedImageError

DEFAULT_MODEL_NAME = "sentence-transformers/clip-ViT-B-32"
SUPPORTED_EXTENSIONS = frozenset({".jpg" , ".jpeg" , ".png" , ".webp" , ".bmp"})


@dataclass(frozen=True)
class SearchResult:
    """One ranked image search result."""

    path: Path
    score: float


def discover_images(image_directory: Path | str) -> list[Path]:
    """Return supported image files below a directory in stable order."""
    directory = Path(image_directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {directory}")

    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def load_image(path: Path | str) -> Image.Image:
    """Load an image, apply its EXIF orientation, and return an RGB copy."""
    with Image.open(path) as source:
        source.load()
        return ImageOps.exif_transpose(source).convert("RGB")


def _batches(items: list[Path] , size: int) -> Iterable[list[Path]]:
    for start in range(0 , len(items) , size):
        yield items[start : start + size]


class ImageSearchEngine:
    """An in-memory CLIP image index that supports natural-language search."""

    def __init__(
        self ,
        model: Any ,
        image_paths: list[Path] ,
        image_embeddings: torch.Tensor ,
    ) -> None:
        if not image_paths:
            raise ValueError("The image index cannot be empty")
        if image_embeddings.ndim != 2:
            raise ValueError("Image embeddings must be a two-dimensional tensor")
        if len(image_paths) != image_embeddings.shape[0]:
            raise ValueError("Each image must have exactly one embedding")

        self.model = model
        self.image_paths = image_paths
        self.image_embeddings = image_embeddings

    @classmethod
    def from_directory(
        cls ,
        image_directory: Path | str ,
        model_name: str = DEFAULT_MODEL_NAME ,
        device: str | None = None ,
        batch_size: int = 32 ,
        model: Any | None = None ,
    ) -> "ImageSearchEngine":
        """Load a model and build an index from all decodable images."""
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")

        paths = discover_images(image_directory)
        if not paths:
            raise ValueError(f"No supported images found in {image_directory}")

        if model is None:
            from sentence_transformers import SentenceTransformer

            selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            model = SentenceTransformer(model_name , device=selected_device)

        valid_paths: list[Path] = []
        embedding_batches: list[torch.Tensor] = []

        for path_batch in _batches(paths , batch_size):
            images: list[Image.Image] = []
            batch_paths: list[Path] = []
            for path in path_batch:
                try:
                    images.append(load_image(path))
                    batch_paths.append(path)
                except (OSError , UnidentifiedImageError) as error:
                    warnings.warn(f"Skipping unreadable image {path}: {error}" , stacklevel=2)

            if not images:
                continue

            try:
                embeddings = model.encode(
                    images ,
                    batch_size=batch_size ,
                    convert_to_tensor=True ,
                    normalize_embeddings=True ,
                    show_progress_bar=False ,
                )
            finally:
                for image in images:
                    image.close()

            if embeddings.ndim == 1:
                embeddings = embeddings.unsqueeze(0)
            embedding_batches.append(embeddings)
            valid_paths.extend(batch_paths)

        if not embedding_batches:
            raise ValueError(f"No decodable images found in {image_directory}")

        return cls(model , valid_paths , torch.cat(embedding_batches , dim=0))

    def search(self , query: str , top_k: int = 5) -> list[SearchResult]:
        """Return images ranked by cosine similarity to a text query."""
        query = query.strip()
        if not query:
            raise ValueError("Enter a search query")

        top_k = int(top_k)
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        top_k = min(top_k , len(self.image_paths))

        query_embedding = self.model.encode(
            query ,
            convert_to_tensor=True ,
            normalize_embeddings=True ,
            show_progress_bar=False ,
        )
        query_embedding = query_embedding.to(self.image_embeddings.device)
        similarities = query_embedding @ self.image_embeddings.T
        scores , indices = similarities.topk(top_k)

        return [
            SearchResult(self.image_paths[index.item()] , score.item())
            for score , index in zip(scores , indices)
        ]
