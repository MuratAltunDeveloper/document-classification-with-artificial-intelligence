import torch
from torchvision.models import resnext101_32x8d

# Pretrained model yükle
model = resnext101_32x8d(pretrained=True)
model.eval()

# Örnek kullanım
input_tensor = torch.rand(1, 3, 224, 224)  # Batch size: 1, RGB channels: 3, Image size: 224x224
output = model(input_tensor)

print(output)
