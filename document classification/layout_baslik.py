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

        #metin basligini bulma
     
        try:
           custom_config = r'--tessdata-dir "C:\Users\murat\OneDrive\Masaüstü\akillilab\Tesseract-OCR\tessdata"'
           metin_baslik = pytesseract.image_to_string(blurred, lang='eng', config=custom_config)

           print(metin_baslik)
        except Exception as e:
            print("hata iç1:",repr(e))

         # Görüntüyü göster
        cv2.imshow("Contoured Image", img_cv)
        cv2.waitKey(0)
        cv2.destroyAllWindows()    
except Exception as e:
    print("Hata:", e)        


