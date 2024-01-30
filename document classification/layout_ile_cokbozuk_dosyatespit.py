import cv2
import numpy as np
from PIL import Image
import pytesseract
def resmin_yuzdekacısiyah(resim):
    toplam_piksel = resim.size
    siyah_piksel = np.sum(resim == 0)
    siyahlikoran=(siyah_piksel / float(toplam_piksel)) * 100
    return siyahlikoran 
def resmin_yuzdekacibeyaz(resim):
    toppixel=resim.size
    beyazpixsel=np.sum(resim==255)
    beyazlikoran = (beyazpixsel / float(toppixel)) * 100
    return beyazlikoran
# Set the path to the Tesseract executable (update with your Tesseract installation path)
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\murat\OneDrive\Masaüstü\akillilab\Tesseract-OCR\tesseract.exe'

# .tif dosyasının yolu (raw string literal kullanılarak)


#sıkıntısız örnek
#tif_file_path = r'C:\Users\murat\OneDrive\Masaüstü\akillilab\data\questionnaire\0000168347.tif'
#sıkıntılı örnek
tif_file_path = r'C:\Users\murat\OneDrive\Masaüstü\akillilab\data\handwritten\500926577_500926582.tif'



# Tif dosyasını aç
try:
    with Image.open(tif_file_path) as img:
        # Görüntüyü OpenCV formatına dönüştür
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
        # Görüntüyü gri tonlamaya çevir
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        
        # Gürültüyü azaltmak için blurla
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

       
        # Siyah renk için alt ve üst sınırları tanımla
        lower_black = np.array([0, 0, 0], dtype=np.uint8)
        upper_black = np.array([30, 30, 30], dtype=np.uint8)

        # Görüntü üzerinde siyah renkteki nesneleri tespit et
        black_mask = cv2.inRange(img_cv, lower_black, upper_black)

        # Siyah renkteki nesneleri siyaha boyayarak vurgula
        img_cv[black_mask != 0] = [0, 255, 0]  #başka renğe boya
        kont=resmin_yuzdekacısiyah(img_cv)
        kont2=resmin_yuzdekacibeyaz(img_cv)
        if(kont>95):
            print("sorun var çok siyah")
        else:
            print("sorun yok siyah")    
        if(kont2>95
           ):
            print("sorun var çok beyaz")
        else:
            print("sorun yok beyaz")       
        # Görüntüyü göster
        cv2.imshow("Orjinal Image", img_cv)
       
        cv2.imshow("Benim siyahı başka renğe boyadığım Image", blurred)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

except Exception as e:
    print("Hata:", e)


 