"""Gradio application for natural-language image search."""

from __future__ import annotations

import argparse
from pathlib import Path

import gradio as gr

from image_search import DEFAULT_MODEL_NAME , ImageSearchEngine

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_IMAGE_DIRECTORY = PROJECT_ROOT / "image_database" / "images"


def create_demo(engine: ImageSearchEngine) -> gr.Interface:
    """Create the Gradio interface around an initialized search engine."""

    def gradio_search(query: str , top_k: int) -> list[tuple[str , str]]:
        return [
            (str(result.path) , f"{result.path.name} - Similarity: {result.score:.3f}")
            for result in engine.search(query , top_k)
        ]

    return gr.Interface(
        fn=gradio_search ,
        inputs=[
            gr.Textbox(label="Search" , placeholder="Try: a dog playing outside") ,
            gr.Slider(
                minimum=1 ,
                maximum=10 ,
                value=min(5 , len(engine.image_paths)) ,
                step=1 ,
                label="Number of Results" ,
            ) ,
        ] ,
        outputs=gr.Gallery(label="Search Results" , columns=5 , height="auto") ,
        title="AI Image Search Engine" ,
        description="""
Search through images using natural language.

Try queries such as **a dog outside**, **something you would eat**,
**a relaxing place**, or **something red**.
""" ,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the AI image search web app")
    parser.add_argument("--image-dir" , type=Path , default=DEFAULT_IMAGE_DIRECTORY)
    parser.add_argument("--model" , default=DEFAULT_MODEL_NAME)
    parser.add_argument("--device" , choices=("cpu" , "cuda" , "mps") , default=None)
    parser.add_argument("--batch-size" , type=int , default=32)
    parser.add_argument("--host" , default="127.0.0.1")
    parser.add_argument("--port" , type=int , default=7860)
    parser.add_argument("--share" , action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Building image index from {args.image_dir}...")
    engine = ImageSearchEngine.from_directory(
        args.image_dir ,
        model_name=args.model ,
        device=args.device ,
        batch_size=args.batch_size ,
    )
    print(f"Indexed {len(engine.image_paths)} images")
    create_demo(engine).launch(
        server_name=args.host ,
        server_port=args.port ,
        share=args.share ,
    )


if __name__ == "__main__":
    main()
