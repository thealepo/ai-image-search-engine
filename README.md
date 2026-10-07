# AI Image Search Engine

A natural-language image search application powered by the
[`clip-ViT-B-32`](https://huggingface.co/sentence-transformers/clip-ViT-B-32)
Sentence Transformers model and Gradio. The included collection contains 398
freely licensed images with source and attribution details in
[`image_database/metadata.csv`](image_database/metadata.csv).

<img width="2060" height="1155" alt="Image" src="https://github.com/user-attachments/assets/8dcd23cc-3f69-4166-a28f-b4cbfaaef2d0" />

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

## Resources

- Colab Notebook: https://colab.research.google.com/drive/1TDNJ_PNq793HlyJQXMhYtL3hHIMGPip_?usp=sharing
- Canva Slides: https://canva.link/tn13wnjnotqcvqr
