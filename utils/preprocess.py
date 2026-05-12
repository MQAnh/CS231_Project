from torchvision import transforms
from .config import IMAGE_SIZE



def preprocess_image(image, model_name):
    if model_name == "MobileNetV2" : 
        image = image.convert("RGB")
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        image_tensor = transform(image)
        image_tensor = image_tensor.unsqueeze(0)
        return image_tensor
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
        
        image_tensor = transform(image)
        image_tensor = image_tensor.unsqueeze(0)
        return image_tensor
