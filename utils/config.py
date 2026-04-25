import torch

MODEL_PATH = r".\models\mobilenetv2.pth"
CLASS_PATH = "class_names.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 224