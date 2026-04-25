from utils.model_loader import load_model, load_class_names
from utils.predictor import predict
from ui.interface import create_interface


class_names = load_class_names()
model = load_model(num_classes=len(class_names))


def predict_fn(image):
    if image is None:
        return {"Vui lòng upload ảnh trước": 1.0}

    return predict(
        image=image,
        model=model,
        class_names=class_names,
        top_k=5
    )


demo = create_interface(predict_fn)

if __name__ == "__main__":
    demo.launch()