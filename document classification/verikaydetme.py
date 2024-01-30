import requests
import json

# rvl-cdip etiket dosyasını alın
rvl_cdip_labels_url = "https://raw.githubusercontent.com/otndata/rvl-cdip/master/labels/labels.json"
response = requests.get(rvl_cdip_labels_url)

if response.status_code == 200:
    with open('labels.json', 'w') as f:
        f.write(response.text)
else:
    print(f"HTTP isteği başarısız. Durum kodu: {response.status_code}")
