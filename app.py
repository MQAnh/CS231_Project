from utils.model_loader import load_class_names, load_all_models
from utils.predictor import predict
from ui.interface import create_interface


class_names = load_class_names()
models_dict = load_all_models(num_classes=len(class_names))


def predict_fn(image, model_name):
    if image is None:
        return {"Vui lòng upload ảnh trước": 1.0}

    model = models_dict[model_name]

    return predict(
        image=image,
        model=model,
        class_names=class_names,
        model_name=model_name, 
        top_k=5, 
    )


demo = create_interface(
    predict_fn=predict_fn,
    model_names=list(models_dict.keys())
)


if __name__ == "__main__":
    demo.launch()