import torch
import torchvision.models as models
from PIL import Image
from torchvision import transforms




# ResNeXt-50 modelini yükle
resnext50 = models.resnext50_32x4d(pretrained=True)
resnext50.eval()
#görselleştirme

def load_and_preprocess_image(image_path):
    image = Image.open(image_path)
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)  # Batch boyutu ekleyin
    return input_batch


def classify_documentation_image(model, image_path):
    input_batch = load_and_preprocess_image(image_path)

    with torch.no_grad():
        output = model(input_batch)

    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    predicted_class = torch.argmax(probabilities).item()

    return predicted_class, probabilities[predicted_class].item()

# Örnek kullanım
image_path =  r'C:\Users\murat\OneDrive\Masaüstü\akillilab\image.png'
predicted_class, confidence = classify_documentation_image(resnext50, image_path)
print(f"Sınıf: {predicted_class}, Güven: {confidence:.4f}")



