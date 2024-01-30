import torch
from torchvision import transforms
from PIL import Image
import json
import timm
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























# Xception modelini indirme ve yerelde saklama kısmı
model_path = r'C:\Users\murat\OneDrive\Masaüstü\akillilab\deneme1\xception_model.pth'   
if not os.path.exists(model_path):
    # Modeli indir
    xception_model = timm.create_model("xception", pretrained=True)
    
    # Modeli yerelde kaydet
    torch.save(xception_model.state_dict(), model_path)
else:
    # Varolan dosyadan modeli yükle
    xception_model = timm.create_model("xception", pretrained=False)
    xception_model.load_state_dict(torch.load(model_path))
xception_model.eval()

# ImageNet sınıf etiketlerini içeren dosyayı al
# Dosyanın bulunduğu dizini doğru bir şekilde belirtin
with open(r'C:\Users\murat\OneDrive\Masaüstü\akillilab\imagenet.txt', 'r') as f:
    labels = [label.strip() for label in f.readlines()]

# Etiketleri bir sözlük formatında oluşturun
imagenet_classes = {str(index): label for index, label in enumerate(labels)}

# 'imagenet_classes.json' dosyasına yazın
with open('imagenet_classes.json', 'w') as f:
    json.dump(imagenet_classes, f)

def classify_image_with_xception(model, image_path, class_labels):
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
    print("Model çıktısı shape:", output.shape)

    # Eğer çıktı birden fazla sınıf skoru içeriyorsa, uygun bir sınıflandırma metodu kullanılmalıdır.
    # Aşağıda argmax kullanarak örnek bir sınıflandırma metodu gösterilmiştir.
    predicted_class = torch.argmax(output, dim=1).item()
    predicted_class_name = class_labels[str(predicted_class)]

    return predicted_class, predicted_class_name
#sınıflandırma
say=0
for jpg_file_path in jpg_file_path_dizisi:
    predicted_class, predicted_class_name = classify_image_with_xception(xception_model, jpg_file_path, imagenet_classes)
    model_etiketler.append(predicted_class_name)
    if "envelope" in predicted_class_name.lower():  # Case-insensitive kontrol için lower kullanıldı
        say += 1
    print(f" Class Name: {predicted_class_name}")

print(f"Toplam {say} adet 'envelope' içeren etiket bulundu.")