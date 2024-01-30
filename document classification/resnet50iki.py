import torch
from torchvision import models, transforms
from PIL import Image

# ResNet-50 modelini yükle
resnet50 = models.resnet50(pretrained=True)
resnet50.eval()

def classify_documentation_image(model, image_path):
    image = Image.open(image_path)

    # Grayscale dönüşümü ekleyin
    preprocess = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),  # 3 kanallı grayscale'e dönüştür
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)  # Batch boyutu ekleyin

    with torch.no_grad():
        output = model(input_batch)

    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    predicted_class = torch.argmax(probabilities).item()

    return predicted_class, probabilities[predicted_class].item()

# Örnek kullanım
image_path = r'C:\Users\murat\OneDrive\Masaüstü\akillilab\image.png'
predicted_class, confidence = classify_documentation_image(resnet50, image_path)
print(f"Sınıf: {predicted_class}, Güven: {confidence:.4f}")
