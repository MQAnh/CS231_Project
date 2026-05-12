import json
import torch
from torchvision import models
from networks.VGG16_bilinear import BilinearVGG16

from .config import CLASS_PATH, DEVICE, MODEL_CONFIGS


def load_class_names():
    with open(CLASS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def build_model(architecture: str, num_classes: int):
    if architecture == "mobilenet_v2":
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = torch.nn.Linear(
            model.classifier[1].in_features,
            num_classes
        )

    elif architecture == "VGG16_Bilinear":
        model = BilinearVGG16(num_classes=100)

    else:
        raise ValueError(f"Unsupported architecture: {architecture}")

    return model


def load_model(model_name: str, num_classes: int):
    config = MODEL_CONFIGS[model_name]

    model = build_model(
        architecture=config["architecture"],
        num_classes=num_classes
    )

    state_dict = torch.load(config["path"], map_location=DEVICE)
    model.load_state_dict(state_dict)

    model.to(DEVICE)
    model.eval()

    return model


def load_all_models(num_classes: int):
    loaded_models = {}

    for model_name in MODEL_CONFIGS.keys():
        loaded_models[model_name] = load_model(
            model_name=model_name,
            num_classes=num_classes
        )

    return loaded_models