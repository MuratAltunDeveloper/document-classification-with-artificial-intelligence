import cv2
import numpy as np
from PIL import Image
import pytesseract
# Set the path to the Tesseract executable (update with your Tesseract installation path)
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\murat\OneDrive\Masaüstü\akillilab\Tesseract-OCR\tesseract.exe'

# .tif dosyasının yolu (raw string literal kullanılarak)
tif_file_path = r'C:\Users\murat\OneDrive\Masaüstü\akillilab\data\budget\00004421_00004422.tif'

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

        # Kontürleri bul
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Büyük dikdörtgenleri filtrele
        large_rectangles = [cnt for cnt in contours if cv2.contourArea(cnt) > 1000]

        # Her bir dikdörtgen için sınırları çiz ve kırmızı renkte vurgula
        for rectangle in large_rectangles:
            x, y, w, h = cv2.boundingRect(rectangle)
            cv2.rectangle(img_cv, (x, y), (x + w, y + h), (0, 100, 255), 2)  # Kırmızı renkte sınırları çiz

        # Görüntüyü göster
        cv2.imshow("Contoured Image", img_cv)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

except Exception as e:
    print("Hata:", e)
