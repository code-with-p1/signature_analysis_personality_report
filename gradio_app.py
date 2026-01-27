import os
os.environ.pop("TF_USE_LEGACY_KERAS", None)

import gradio as gr
from common.gradio.common import full_analysis

with gr.Blocks(theme=gr.themes.Soft()) as demo:

    gr.Markdown("# Handwriting → Big Five Personality Prediction")
    gr.Markdown("Upload any image of handwriting → model will try to predict personality trait")

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(
                type="pil",
                label="Upload handwriting image",
                sources=["upload"],
                height=380
            )

        with gr.Column():
            gr.Markdown("### Prediction")
            prediction_output = gr.Markdown(value="Upload image and click Analyze...")

            gr.Markdown("### Personality Description")
            summary_output = gr.Markdown(value="Description will appear here...")

    btn = gr.Button("Analyze", variant="primary")
    btn.click(
        fn=full_analysis,
        inputs=image_input,
        outputs=[prediction_output, summary_output]
    )

    image_input.change(
        fn=full_analysis,
        inputs=image_input,
        outputs=[prediction_output, summary_output]
    )


if __name__ == "__main__":
    demo.launch(
        # server_name="0.0.0.0",
        # server_port=7860
        share=True,
        debug=True,
        inbrowser=True,
    )