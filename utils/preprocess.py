from torchvision import transforms
from .config import IMAGE_SIZE


transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])


def preprocess_image(image):
    image = image.convert("RGB")
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)
    return image_tensor