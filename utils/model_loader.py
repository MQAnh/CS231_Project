import json
import torch
from torchvision import models
from .config import MODEL_PATH, CLASS_PATH, DEVICE


def load_class_names():
    with open(CLASS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_model(num_classes: int):
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = torch.nn.Linear(
        model.classifier[1].in_features,
        num_classes
    )

    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)

    model.to(DEVICE)
    model.eval()

    return model