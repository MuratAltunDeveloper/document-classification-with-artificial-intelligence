import torch
from torchvision import models, transforms
from PIL import Image
import json
import requests
from io import BytesIO
from termcolor import colored
def preprocess_image(image, target_size=(224, 224)):
    preprocess = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return preprocess(image)

def classify_image_with_resnet50(model, image_path, preprocess, class_labels):
    image = Image.open(image_path)
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)

    with torch.no_grad():
        output = model(input_batch)

    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    predicted_class = torch.argmax(probabilities).item()
    predicted_class_name = class_labels[str(predicted_class)]

    return predicted_class, predicted_class_name, probabilities[predicted_class].item()

# Örnek kullanım
resnet50 = models.resnet50(pretrained=True)
resnet50.eval()

# rvl-cdip etiket dosyasını alın
rvl_cdip_labels_url = "https://raw.githubusercontent.com/otndata/rvl-cdip/master/labels/labels.json"
rvl_cdip_labels = requests.get(rvl_cdip_labels_url).json()

# Örnek bir resim indirin (örnek olarak ilk resmi kullanıyoruz)
image_path = r'C:\Users\murat\OneDrive\Masaüstü\akillilab\image.jpg'
image = Image.open(image_path)

# Sınıf tahmini yapın
predicted_class, predicted_class_name, confidence = classify_image_with_resnet50(resnet50, image, preprocess_image, rvl_cdip_labels)

# Renkli çıktıyı yazdırın
# ANSI renk kodları kullanarak renkli çıktı
print(f"Sınıf: {predicted_class} {predicted_class_name}, Güven: {confidence:.4f}")