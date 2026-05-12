import gradio as gr


def create_interface(predict_fn, model_names):
    with gr.Blocks(title="Butterfly Classification Demo") as demo:
        gr.Markdown(
            """
            # Butterfly Classification Demo

            Upload ảnh bướm, chọn model và dự đoán loài bướm tương ứng.
            """
        )

        with gr.Row():
            with gr.Column():
                image_input = gr.Image(
                    type="pil",
                    label="Upload ảnh bướm"
                )

                model_dropdown = gr.Dropdown(
                    choices=model_names,
                    value=model_names[0],
                    label="Chọn model"
                )

                predict_button = gr.Button("Dự đoán")
                clear_button = gr.Button("Xóa ảnh / nhập ảnh mới")

            with gr.Column():
                label_output = gr.Label(
                    num_top_classes=5,
                    label="Kết quả dự đoán"
                )

        predict_button.click(
            fn=predict_fn,
            inputs=[image_input, model_dropdown],
            outputs=label_output
        )

        clear_button.click(
            fn=lambda: (None, None),
            inputs=None,
            outputs=[image_input, label_output]
        )

    return demo