import os
import requests
from pathlib import Path
from PIL import Image
from io import BytesIO
import numpy as np
from feature_extractor import FeatureExtractor
from pymongo import MongoClient
from flask_cors import CORS
from pymongo.collection import Collection

class FeatureExtractorManager:
    def __init__(self):
        # self.connection_string = os.environ.get("mongodb+srv://minhkietpeople:IvptaLQLohowN4aE@cluster0-minhkietwebeco.ovhz23k.mongodb.net/?retryWrites=true&w=majority")
        # self.mongo_client = MongoClient(self.connection_string)
        # self.database = self.mongo_client.get_database("test")
        # self.collection = Collection = self.database.get_collection("products")
        self.fe = FeatureExtractor()
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
        # self.delete_feature_files()
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
                print("Thông số:", feature)

