"""Hugging Face Spaces Entrypoint for Neev AI Facade Screening Platform."""

import os
import uvicorn
from neev.app.main import app as fastapi_app

# If gradio is installed (on Hugging Face Spaces), mount Gradio for SDK compliance
try:
    import gradio as gr
    demo = gr.Blocks(title="Neev: AI Building Facade Defect Screening")
    app = gr.mount_gradio_app(fastapi_app, demo, path="/_gradio")
except ImportError:
    demo = None
    app = fastapi_app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    host = os.environ.get("HOST", "0.0.0.0")
    uvicorn.run("app:app", host=host, port=port)
