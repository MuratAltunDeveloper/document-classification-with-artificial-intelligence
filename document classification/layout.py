'''GENEL KOD HATA ALIRSAN BUNU KULLAN
*ilerleyiş  layout ile neyi çekeceğim
TABLOLAR
GRAFİKLER
RESİMLER 
SEMALAR VE DİYAGRAMLAR
ÜSTTEKİ BOŞLUK



































import cv2
import numpy as np
from PIL import Image
import pytesseract

# Set the path to the Tesseract executable (update with your Tesseract installation path)
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\murat\OneDrive\Masaüstü\akillilab\Tesseract-OCR\tesseract.exe'


# .tif dosyasının yolu (raw string literal kullanılarak)
tif_file_path = r'C:\Users\murat\OneDrive\Masaüstü\akillilab\data\memo\0000006343.tif'

# Tif dosyasını aç
try:
    with Image.open(tif_file_path) as img:
        # Görüntüyü OpenCV formatına dönüştür
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
        # Görüntüyü gri tonlamaya çevir
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        
        # Gürültüyü azaltmak için blurla
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Kenarları tespit et
        edges = cv2.Canny(blurred, 50, 150)
        
        # Tesseract OCR ile metni çıkart
        try:
         metin = pytesseract.image_to_string(blurred, lang='eng')
        except Exception as e:
         print(repr(e))
        # Konturları bul
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Konturları çiz
        cv2.drawContours(img_cv, contours, -1, (0, 254, 0), 2)
        
        # Görüntüyü göster
        cv2.imshow("Contoured Image", img_cv)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

except Exception as e:
    print("Hata:", e)
'''