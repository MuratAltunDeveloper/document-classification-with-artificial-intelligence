import torch
from torchvision import models, transforms
from PIL import Image
import json
import requests
from io import BytesIO
from termcolor import colored
import os
def convert_tif_to_jpeg(tif_path, jpeg_path):
    # .tif dosyasını aç
    tif_image = Image.open(tif_path)

    # .jpeg olarak kaydet
    jpeg_image_path, _ = os.path.splitext(jpeg_path)
    jpeg_image_path += ".jpg"
    tif_image.save(jpeg_image_path, format='JPEG')

def list_files_in_directory(directory):
    file_list = []
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            file_list.append(file_path)
    return file_list

# .tif dosyalarının yolu
tif_file_path_dizisi = []

# Örnek kullanım
directory_path = "C:\\Users\\murat\\OneDrive\\Masaüstü\\akillilab\\data\\email"
files = list_files_in_directory(directory_path)

art = 0
for file_path in files:
    tif_file_path_dizisi.append(file_path)
    art = art + 1
    if art == 300:
        break

# .jpg dosyalarının yolu
jpg_file_path_dizisi = []

# ImageNet sınıf etiketlerini içeren dosyayı al
# Dosyanın bulunduğu dizini doğru bir şekilde belirtin
with open(r'C:\Users\murat\OneDrive\Masaüstü\akillilab\imagenet.txt', 'r') as f:
    labels = [label.strip() for label in f.readlines()]

# Etiketleri bir sözlük formatında oluşturun
imagenet_classes = {str(index): label for index, label in enumerate(labels)}

# 'imagenet_classes.json' dosyasına yazın
with open('imagenet_classes.json', 'w') as f:
    json.dump(imagenet_classes, f)
#resnet50
resnet50 = models.resnet50(pretrained=True)
resnet50.eval()
output_folder = r'C:\Users\murat\OneDrive\Masaüstü\resim'
# .tif dosyalarını .jpg'ye dönüştür ve yeni .jpg dosyalarının yollarını ata
for i, tif_file_path in enumerate(tif_file_path_dizisi):
    jpg_file_path = os.path.join(output_folder, f"resim{i + 1}.jpg")
    convert_tif_to_jpeg(tif_file_path, jpg_file_path)
    jpg_file_path_dizisi.append(jpg_file_path)

# Gerçek ve Model Etiketleri
real_etiketler = []
model_etiketler = []


def classify_images_with_xception(model, image_paths, class_labels):
    predictions = []

    for image_path in image_paths:
        # Görüntüyü RGB olarak aç
        image = Image.open(image_path).convert("RGB")

        preprocess = transforms.Compose([
            transforms.Resize((299, 299)),  # Xception için önerilen boyut
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        input_tensor = preprocess(image)
        input_batch = input_tensor.unsqueeze(0)

        with torch.no_grad():
            output = model(input_batch)

        # Çıktıdaki boyutlarını kontrol et
        #print("Model çıktısı shape:", output.shape)

        # Eğer çıktı birden fazla sınıf skoru içeriyorsa, uygun bir sınıflandırma metodu kullanılmalıdır.
        # Aşağıda argmax kullanarak örnek bir sınıflandırma metodu gösterilmiştir.
        predicted_class = torch.argmax(output, dim=1).item()
        predicted_class_name = class_labels[str(predicted_class)]
        predictions.append(predicted_class_name)

        print(f" Class Name: {predicted_class_name}")

    return predictions

# Örnek kullanım
model_etiketler = classify_images_with_xception(resnet50, jpg_file_path_dizisi, imagenet_classes)

# 'envelope' içeren etiket sayısını kontrol et
say = sum("envelope" in label.lower() for label in model_etiketler)
print(f"Toplam {say} adet 'envelope' içeren etiket bulundu.")
