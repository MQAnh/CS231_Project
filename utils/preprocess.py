from torchvision import transforms
from .config import IMAGE_SIZE
from torchvision.models import AlexNet_Weights



def preprocess_image(image, model_name):
    transform = None
    if model_name == "MobileNetV2" : 
        image = image.convert("RGB")
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
    elif model_name == "VGG16_Bilinear" : 
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ])
    elif model_name == "Resnet50" or model_name == "Googlenet": 
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225]
                )
            ]
        )
    elif model_name == "Alexnet" : 
        weights = AlexNet_Weights.IMAGENET1K_V1
        imagenet_mean = weights.transforms().mean
        imagenet_std = weights.transforms().std

        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ])

        
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)
    return image_tensor
