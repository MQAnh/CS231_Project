import gradio as gr


def create_interface(predict_fn):
    with gr.Blocks(title="Butterfly Classification Demo") as demo:
        gr.Markdown(
            """
            # 🦋 Butterfly Classification Demo
            Upload ảnh bướm, model sẽ dự đoán loài bướm tương ứng.
            Sau khi có kết quả, bạn có thể upload ảnh khác và dự đoán tiếp.
            """
        )

        with gr.Row():
            with gr.Column():
                image_input = gr.Image(
                    type="pil",
                    label="Upload ảnh bướm"
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
            inputs=image_input,
            outputs=label_output
        )

        clear_button.click(
            fn=lambda: (None, None),
            inputs=None,
            outputs=[image_input, label_output]
        )

    return demo