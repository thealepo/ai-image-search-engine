# AI Explore Image Search Dataset

This ready-to-use collection contains **398 images** for the beginner workshop **“AI Explore: Build Your Own AI Image Search Engine.”** It is designed for open-vocabulary retrieval with a pretrained CLIP-style model. The final count is intentionally just under 400 after strict visual, provenance, and retrieval-quality review; weak entries were not retained merely to hit a round number.

This is not a classification dataset. Images were chosen to form overlapping semantic neighborhoods: a single image may contain a subject, action, setting, color, time of day, social context, and visual mood. Changing one word in a query should therefore rearrange results rather than merely switch between rigid classes.

## Contents

```text
ai_explore_image_database/
├── images/
│   ├── 0001.jpg
│   ├── 0002.jpg
│   └── ...
├── metadata.csv
├── example_queries.txt
└── README.md
```

The application can simply iterate through `images/`. The numeric filenames are stable identifiers, not labels.

## Sources, licenses, and attribution

All images were downloaded from **Wikimedia Commons** on **2026-10-02**. Commons is a source repository, not a single blanket license: each image retains its own license or public-domain status.

The collection contains the following source-reported license labels:

| License | Images |
|---|---:|
| CC BY-SA 4.0 | 158 |
| CC BY 2.0 | 48 |
| CC BY-SA 3.0 | 44 |
| CC0 | 42 |
| Public domain | 35 |
| CC BY-SA 2.0 | 29 |
| CC BY 4.0 | 28 |
| CC BY 3.0 | 11 |
| CC BY 2.5 | 2 |
| CC BY-SA 1.0 | 1 |

`metadata.csv` keeps the creator, exact license label, license URL, Commons file page, original file URL, original title, and source SHA-1 where available. Keep the CSV with the images. Before external publication or redistribution, check each `source_url` for its complete and current credit instructions.

Depending on the image, reuse requirements may include crediting the creator, naming and linking the license, linking the source, indicating modifications, or applying compatible ShareAlike terms to an adaptation. A practical credit pattern is:

> “[Image title or description]” by [creator], [license], via Wikimedia Commons ([source URL]).

See the Wikimedia Commons [reuse guidance](https://commons.wikimedia.org/wiki/Commons:Reusing_content_outside_Wikimedia) and [licensing policy](https://commons.wikimedia.org/wiki/Commons:Licensing). Public-domain status can vary by jurisdiction, and copyright freedom does not eliminate privacy, publicity, trademark, cultural, or other non-copyright considerations. The metadata is an audit aid, not a legal warranty.

## Composition

Primary curation buckets are only a coverage aid; many images bridge several buckets.

| Primary visual story | Images |
|---|---:|
| Landscapes, weather, and nature | 57 |
| People and everyday actions | 55 |
| Animals in context | 51 |
| Architecture, cities, and interiors | 50 |
| Sports and active recreation | 42 |
| Transportation and travel | 41 |
| Food, drink, and cooking | 34 |
| Technology and everyday objects | 28 |
| Plants, flowers, and gardens | 22 |
| Visual and mood wildcards | 18 |

The collection mixes indoor and outdoor scenes, groups and individuals, land/water/air transportation, different weather and times of day, distinctive colors, and visual ideas such as peaceful, cozy, dramatic, crowded, lonely, playful, and exciting. It is primarily photographic, with a small number of freely licensed historical or illustrative images to add visual-style diversity.

The concepts field averages **7.5 concepts per image**; **364 of 398 images** have at least four audit concepts. These concepts are not ground-truth labels and must not be passed into the retrieval model.

## Preprocessing and curation

- Source images were decoded and checked for corruption.
- EXIF orientation was applied.
- Images were converted to RGB JPEG.
- Aspect ratios were preserved and images were not intentionally upscaled.
- The long edge was capped at 1280 pixels; every retained image has a short edge of at least 500 pixels.
- JPEGs were saved progressively at quality 88.
- Exact source duplicates, normalized-pixel duplicates, and close perceptual-hash matches were rejected during collection.
- Sixteen thematic contact sheets were reviewed manually. Weak lexical matches, text-dominated graphics, sensitive material, low-information scenes, and category-name traps were removed.
- The final 398 files occupy about 108 MiB, small enough for a workshop while retaining ample detail for CLIP.

## Metadata

`metadata.csv` contains one row per image. Core columns are:

| Column | Meaning |
|---|---|
| `filename` | Exact local filename inside `images/`. |
| `source` | Source repository. |
| `source_url` | Wikimedia Commons file-description page. |
| `license`, `license_url` | Source-reported license and its link. |
| `author` | Source-reported creator/credit. |
| `short_description` | Concise human-readable visual description. |
| `concepts` | Comma-separated semantic concepts for auditing. |
| `theme` | Broad curation bucket, not a class label. |
| `discovery_query` | Phrase used to discover or manually describe the image. |
| `original_title`, `original_url` | Original Commons identity and file URL. |
| `source_page_id`, `source_sha1` | Source provenance identifiers. |
| `original_width`, `original_height` | Source dimensions reported by Commons. |
| `processed_width`, `processed_height` | Local workshop-image dimensions. |
| `file_sha256` | SHA-256 of the local JPEG. |
| `retrieved_date` | Date the source metadata was collected. |
| `modifications` | Preprocessing applied to the local copy. |
| `attribution_required`, `usage_terms`, `restrictions` | Additional source-reported reuse fields. |

Descriptions and concepts are intended for human inspection only. Retrieval should use only CLIP image embeddings.

## Loading the images

Run code from the dataset root:

```python
from pathlib import Path
from PIL import Image

image_paths = sorted(Path("images").glob("*.jpg"))

for image_path in image_paths:
    with Image.open(image_path) as source_image:
        image = source_image.convert("RGB")
        # Pass `image` to the model's image preprocessor.
        # Keep `image_path.name` to map results back to metadata.csv.
```

`example_queries.txt` contains 40 workshop-friendly searches, ranging from simple objects to combinations of subject, action, setting, color, and mood.

## Validation

The final validation checks:

- every image can be fully decoded;
- filenames and metadata rows correspond exactly;
- required provenance fields are present;
- recorded dimensions and SHA-256 values match the files;
- exact duplicates are absent; and
- close pHash/dHash duplicate candidates are reported for review.

The dataset was also evaluated qualitatively with `openai/clip-vit-base-patch32` using the example queries. CLIP retrieval is model- and wording-dependent, so visual judgment remains more important than a raw similarity score.

## Caveats

- This is a small teaching collection, not a statistically representative sample or benchmark.
- Broad coverage is intentional; counts are not perfectly balanced classes.
- Human descriptions and concepts can contain subjective judgments or occasional errors.
- CLIP-style models can reproduce social and cultural biases from their training data.
- Fine-grained identity, text reading, counting, and subtle spatial relationships may be unreliable.
- Recheck the linked Commons page before external redistribution because source metadata or licensing notes can change.

