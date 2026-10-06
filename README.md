# AI Image Search Engine

A natural-language image search application powered by the
[`clip-ViT-B-32`](https://huggingface.co/sentence-transformers/clip-ViT-B-32)
Sentence Transformers model and Gradio. The included collection contains 398
freely licensed images with source and attribution details in
[`image_database/metadata.csv`](image_database/metadata.csv).

## How it works

At startup, the application:

1. Finds and validates images in `image_database/images`.
2. Embeds them in batches with CLIP.
3. Embeds each text query in the same vector space.
4. Ranks images by cosine similarity.

Image and text embeddings are normalized, so their dot product is cosine
similarity.

## Requirements

- Python 3.10 or newer
- Up to 7 GB of free disk space for Python packages and the downloaded model
- Internet access on the first run to download the model from Hugging Face

A CUDA GPU is optional. The app automatically uses CUDA when available and
otherwise runs on the CPU.

## Run locally

```bash
git clone https://github.com/thealepo/ai-image-search-engine.git
cd ai-image-search-engine
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

Open <http://127.0.0.1:7860>. Initial startup can take a few minutes while the
model downloads and the 398 images are embedded.

Useful options:

```bash
python app.py --help
python app.py --device cpu
python app.py --host 0.0.0.0 --port 8080
python app.py --image-dir /path/to/other/images
python app.py --share
```

The custom image directory may contain JPEG, PNG, WebP, or BMP files and is
searched recursively. Unreadable files are skipped with a warning.

## Google Colab

Run these commands in a Colab cell after cloning the repository:

```python
%cd /content/ai-image-search-engine
%pip install -q -r requirements.txt
!python app.py --share
```

Gradio prints a temporary public URL. The Python application replaces the
notebook's global state while preserving its search behavior.

## Test

Tests use a deterministic stand-in model, so they do not need network access or
a CLIP download. They also verify that all included images decode successfully
and correspond exactly to the metadata.

```bash
pip install -r requirements-dev.txt
pytest -q
```

GitHub Actions runs the same suite for every push and pull request.

## Project layout

```text
app.py                    Gradio UI and command-line entry point
image_search.py           Image loading, embedding, and ranking logic
image_database/images/    Included search collection
image_database/metadata.csv
requirements.txt          Runtime dependencies
tests/                    Unit and dataset integrity tests
```

## Dataset and responsible use

See [`image_database/README.md`](image_database/README.md) for provenance,
licenses, attribution requirements, preprocessing details, and model caveats.
CLIP results are wording-dependent and may reflect biases in its training data.
