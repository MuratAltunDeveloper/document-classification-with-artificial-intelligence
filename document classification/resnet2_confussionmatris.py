import torch
import os
from torchvision import transforms
from PIL import Image
import json
from torchvision.models import resnext101_32x8d

from PIL import Image
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

output_folder = r'C:\Users\murat\OneDrive\Masaüstü\resim'
# .tif dosyalarını .jpg'ye dönüştür ve yeni .jpg dosyalarının yollarını ata
for i, tif_file_path in enumerate(tif_file_path_dizisi):
    jpg_file_path = os.path.join(output_folder, f"resim{i + 1}.jpg")
    convert_tif_to_jpeg(tif_file_path, jpg_file_path)
    jpg_file_path_dizisi.append(jpg_file_path)

# Gerçek ve Model Etiketleri
real_etiketler = []
model_etiketler = []

# SE-ResNeXt-101(32x4d) modelini yükleme
seresnext101 = resnext101_32x8d(pretrained=True)
seresnext101.eval()

# ImageNet sınıf etiketlerini içeren dosyayı al
with open(r'C:\Users\murat\OneDrive\Masaüstü\akillilab\imagenet.txt', 'r') as f:
    labels = [label.strip() for label in f.readlines()]

# Etiketleri bir sözlük formatında oluşturun
imagenet_classes = {str(index): label for index, label in enumerate(labels)}

# 'imagenet_classes.json' dosyasına yazın
with open('imagenet_classes.json', 'w') as f:
    json.dump(imagenet_classes, f)

# classify_image_with_seresnext101 fonksiyonunu tanımla
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
say = 0
# .jpg dosyalarını kullanarak sınıflandırma yap
for jpg_file_path in jpg_file_path_dizisi:
    predicted_class, predicted_class_name = classify_image_with_seresnext101(seresnext101, jpg_file_path, imagenet_classes)
    model_etiketler.append(predicted_class_name)
    if "envelope" in predicted_class_name.lower():  # Case-insensitive kontrol için lower kullanıldı
        say += 1
for i in model_etiketler:
 print(f"={i}")
 
print(f"Toplam {say} adet 'envelope' içeren etiket bulundu.")
