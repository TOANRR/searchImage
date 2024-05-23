import numpy as np
from PIL import Image
from feature_extractor import FeatureExtractor
from datetime import datetime
from pathlib import Path
from io import BytesIO
from flask_cors import CORS, cross_origin
import os
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
from flask_cors import CORS, cross_origin
import requests
mongo_client: MongoClient = MongoClient("mongodb+srv://minhkietpeople:TNJ99tPbdFhNIbOH@cluster0-minhkiet.bebh5pw.mongodb.net/?retryWrites=true&w=majority")
database: Database = mongo_client.get_database("test")
collection = Collection = database.get_collection("products")
folder_path = './static/feature'

# Lấy danh sách tất cả các tệp trong thư mục
file_list = os.listdir(folder_path)

# Xóa từng tệp một
for file_name in file_list:
    file_path = os.path.join(folder_path, file_name)
    try:
        os.remove(file_path)
    except Exception as e:
        print(f"Không thể xóa tệp {file_name}: {e}")
prods =[]
products= list(collection.find())
for product in products :
   ob = {"id": product['_id'], "images" : product['images']}
   prods.append(ob)
print(prods)

if __name__ == '__main__':
    fe = FeatureExtractor()

    for prod in prods:
        i=0
        for im in prod['images']:
            i=i+1
            file = requests.get(im)
            img = Image.open(BytesIO(file.content))
            img = img.convert('RGB')
            feature = fe.extract(img)

            print ("thong so", feature)
            feature_path = Path("./static/feature") / (str(i)+'.'+str(prod['id']) + ".npy")  # e.g., ./static/feature/xxx.npy
            np.save(feature_path, feature)
   