import torch
import torch.nn.functional as F

from .config import DEVICE
from .preprocess import preprocess_image


def predict(image, model, class_names, model_name = "MobileNetV2", top_k=5):
    image_tensor = preprocess_image(image, model_name).to(DEVICE)

    with torch.no_grad():
        outputs = model(image_tensor)
        probs = F.softmax(outputs, dim=1)[0]

    top_probs, top_indices = torch.topk(probs, k=top_k)

    results = {}

    for prob, idx in zip(top_probs, top_indices):
        label = class_names[str(idx.item())]
        results[label] = prob.item()

    return results