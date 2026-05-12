import torch

CLASS_PATH = "class_names.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 224

MODEL_CONFIGS = {
    "MobileNetV2": {
        "architecture": "mobilenet_v2",
        "path": r".\models\mobilenetv2.pth"
    },
    "VGG16_Bilinear": {
        "architecture": "VGG16_Bilinear",
        "path": r".\models\vgg16_bilinear_lr_0.001.pt"
    },
    "Resnet50": {
        "architecture": "Resnet50",
        "path": r".\models\resnet50.pt"
    },
    "Alexnet": {
        "architecture": "Alexnet",
        "path": r".\models\alexnet.pth"
    },
    "Googlenet": {
        "architecture": "Googlenet",
        "path": r".\models\googlenet.pth"
    },
    
}