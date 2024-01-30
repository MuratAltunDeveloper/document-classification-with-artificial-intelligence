import torch
from torchvision import models, transforms
from PIL import Image
import json

# ResNet-50 modelini yükle
resnet50 = models.resnet50(pretrained=True)
resnet50.eval()

# ImageNet sınıf etiketlerini içeren dosyayı al
# Dosyanın bulunduğu dizini doğru bir şekilde belirtin
with open(r'C:\Users\murat\OneDrive\Masaüstü\akillilab\imagenet.txt', 'r') as f:
    labels = [label.strip() for label in f.readlines()]

# Etiketleri bir sözlük formatında oluşturun
imagenet_classes = {str(index): label for index, label in enumerate(labels)}

# 'imagenet_classes.json' dosyasına yazın
with open('imagenet_classes.json', 'w') as f:
    json.dump(imagenet_classes, f)

def classify_documentation_image(model, image_path, class_labels):
    image = Image.open(image_path)

    preprocess = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)

    with torch.no_grad():
        output = model(input_batch)

    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    predicted_class = torch.argmax(probabilities).item()
    predicted_class_name = class_labels[str(predicted_class)]

    return predicted_class, predicted_class_name, probabilities[predicted_class].item()

# Örnek kullanım
image_path = r'C:\Users\murat\OneDrive\Masaüstü\akillilab\image.png'
predicted_class, predicted_class_name, confidence = classify_documentation_image(resnet50, image_path, imagenet_classes)
print(f"Sınıf: {predicted_class} ({predicted_class_name}), Güven: {confidence:.4f}")
