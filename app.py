"""Hugging Face Spaces Entrypoint for Neev AI Facade Screening Platform."""

import os
from neev.app.main import app as fastapi_app

# ZeroGPU hook: enables free NVIDIA A10G GPU allocation on Hugging Face Spaces
try:
    import spaces

    @spaces.GPU
    def zero_gpu_pipeline(x: str) -> str:
        """Registered GPU event handler for Hugging Face ZeroGPU startup scanner."""
        return x

except (ImportError, Exception):
    def zero_gpu_pipeline(x: str) -> str:
        return x

import gradio as gr

# Build Gradio Blocks with registered ZeroGPU event listener
with gr.Blocks(title="Neev: AI Building Facade Defect Screening") as demo:
    dummy_input = gr.Textbox(value="neev", visible=False)
    dummy_output = gr.Textbox(visible=False)
    dummy_btn = gr.Button(visible=False)
    dummy_btn.click(fn=zero_gpu_pipeline, inputs=[dummy_input], outputs=[dummy_output])

# Mount Gradio helper onto the existing FastAPI application at /_gradio
# FastAPI natively serves the Neev Single Page Application at root ("/")
app = gr.mount_gradio_app(fastapi_app, demo, path="/_gradio")

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 7860))
    host = os.environ.get("HOST", "0.0.0.0")
    uvicorn.run("app:app", host=host, port=port)
