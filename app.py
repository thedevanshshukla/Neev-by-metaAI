"""Hugging Face Spaces Entrypoint for Neev AI Facade Screening Platform."""

import os

# ZeroGPU support: Enables dynamic NVIDIA A10G (24GB VRAM) allocation on Hugging Face Spaces
try:
    import spaces

    @spaces.GPU
    def gpu_warmup():
        """Top-level ZeroGPU hook detected during Hugging Face Spaces startup."""
        return True

except (ImportError, Exception):
    pass

from neev.app.main import app as fastapi_app

try:
    import gradio as gr

    demo = gr.Blocks(title="Neev: AI Building Facade Defect Screening")
    app = gr.mount_gradio_app(fastapi_app, demo, path="/_gradio")
except (ImportError, Exception):
    demo = None
    app = fastapi_app

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 7860))
    host = os.environ.get("HOST", "0.0.0.0")
    uvicorn.run("app:app", host=host, port=port)
