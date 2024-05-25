import os
import requests
from pathlib import Path
from PIL import Image
from io import BytesIO
import numpy as np
from feature_extractor import FeatureExtractor
from feature_extractorVGG16 import FeatureExtractorVGG16
from feature_extractorResnet50V2 import FeatureExtractorResnet50V2
import time


class FeatureExtractorManager:
    def __init__(self):
       
        self.fe = FeatureExtractorResnet50V2()
        self.folder_path = './static/feature'
        

    def delete_feature_files(self):
      
        file_list = os.listdir(self.folder_path)
        for file_name in file_list:
            file_path = os.path.join(self.folder_path, file_name)
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Không thể xóa tệp {file_name}: {e}")

    def extract_features(self,products):
        start_time = time.time()  # Bắt đầu đo thời gian

        self.delete_feature_files()
        prods = []
        for product in products:
            ob = {"id": product['_id'], "images": product['images']}
            prods.append(ob)

        for prod in prods:
            i = 0
            for im in prod['images']:
                i += 1
                file = requests.get(im)
                img = Image.open(BytesIO(file.content))
                img = img.convert('RGB')
                feature = self.fe.extract(img)
                feature_path = Path(f"{self.folder_path}/{i}.{prod['id']}.npy")
                np.save(feature_path, feature)
                # print("Thông số:", feature)

        end_time = time.time()  # Kết thúc đo thời gian
        processing_time = end_time - start_time
        print(f"Thời gian xử lý: {processing_time:.2f} giây")

