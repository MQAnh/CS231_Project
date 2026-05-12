import torch
import torch.nn as nn 
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim

import torchvision.models as models
import torchvision.transforms as transforms

class BilinearVGG16(nn.Module):
    def __init__(self, num_classes):
        super(BilinearVGG16, self).__init__()
        
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)


        self.features = vgg.features
    
        for param in self.features.parameters():
            param.requires_grad = False

        self.classifier = nn.Linear(512 * 512, num_classes)
        
    def forward(self, x):

        x = self.features(x)
        
        batch_size, channels, height, width = x.size()
        
        x = x.view(batch_size, channels, height * width)
        
        # Outer product ([B, C, H*W] * [B, H*W, C] -> [B, C, C])
        bilinear_matrix = torch.bmm(x, x.transpose(1, 2))
        
        # Flatten [B, C * C]
        v = bilinear_matrix.view(batch_size, channels * channels)
        
        # Element-wise square root
        v = torch.sign(v) * torch.sqrt(torch.abs(v) + 1e-12)
        
        # L2 Norm
        v = nn.functional.normalize(v, p=2, dim=1)
        
        out = self.classifier(v)
        return out