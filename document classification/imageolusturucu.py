from PIL import Image

def convert_tif_to_jpg(input_path, output_path):
    # .tif dosyasını yükle
    tif_image = Image.open(input_path)
    
    # .jpg olarak kaydet
    tif_image.save(output_path, format='JPEG')

# Kullanım örneği
tif_file_path = r'C:\Users\murat\OneDrive\Masaüstü\akillilab\data\news article\0000149400.tif'
jpg_file_path = r'C:\Users\murat\OneDrive\Masaüstü\akillilab\image.jpg'

convert_tif_to_jpg(tif_file_path, jpg_file_path)
