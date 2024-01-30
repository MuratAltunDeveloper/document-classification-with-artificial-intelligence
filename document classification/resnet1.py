import torch
from torchvision import transforms
from PIL import Image
import json
from torchvision.models import resnext101_32x8d

# SE-ResNeXt-101(32x4d) modelini yükle
seresnext101 = resnext101_32x8d(pretrained=True)
seresnext101.eval()

# ImageNet sınıf etiketlerini içeren dosyayı al
# Dosyanın bulunduğu dizini doğru bir şekilde belirtin
with open(r'C:\Users\murat\OneDrive\Masaüstü\akillilab\imagenet.txt', 'r') as f:
    labels = [label.strip() for label in f.readlines()]

# Etiketleri bir sözlük formatında oluşturun
imagenet_classes = {str(index): label for index, label in enumerate(labels)}

# 'imagenet_classes.json' dosyasına yazın
with open('imagenet_classes.json', 'w') as f:
    json.dump(imagenet_classes, f)

def classify_image_with_seresnext101(model, image_path, class_labels):
    # Görüntüyü RGB olarak aç
    image = Image.open(image_path).convert("RGB")

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)

    with torch.no_grad():
        output = model(input_batch)

    # Çıktıdaki boyutlarını kontrol et
    print("Model çıktısı shape:", output.shape)

    # Eğer çıktı birden fazla sınıf skoru içeriyorsa, uygun bir sınıflandırma metodu kullanılmalıdır.
    # Aşağıda argmax kullanarak örnek bir sınıflandırma metodu gösterilmiştir.
    predicted_class = torch.argmax(output, dim=1).item()
    predicted_class_name = class_labels[str(predicted_class)]

    return predicted_class, predicted_class_name

# Örnek kullanım
image_path = r'C:\Users\murat\OneDrive\Masaüstü\akillilab\image.png'
predicted_class, predicted_class_name = classify_image_with_seresnext101(seresnext101, image_path, imagenet_classes)
print(f"Sınıf: {predicted_class} ({predicted_class_name})")
