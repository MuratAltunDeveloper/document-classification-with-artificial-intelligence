from PIL import Image

def convert_tif_to_png(input_path, output_path):
    # .tif dosyasını yükle
    tif_image = Image.open(input_path)
    
    # .png olarak kaydet
    tif_image.save(output_path, format='PNG')

# Kullanım örneği
tif_file_path = r'C:\Users\murat\OneDrive\Masaüstü\akillilab\data\news article\0000149400.tif'
png_file_path = r'C:\Users\murat\OneDrive\Masaüstü\akillilab\image.png'

convert_tif_to_png(tif_file_path, png_file_path)
