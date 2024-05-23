import numpy as np
from PIL import Image
from feature_extractor import FeatureExtractor
from prepareCollection import FeatureExtractorManager
from datetime import datetime
from flask import Flask, request, jsonify
from pathlib import Path
from io import BytesIO
from flask_cors import CORS, cross_origin
from json import dumps
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
from flask_cors import CORS, cross_origin
import requests
from sklearn.neighbors import NearestNeighbors

mongo_client: MongoClient = MongoClient("mongodb+srv://minhkietpeople:TNJ99tPbdFhNIbOH@cluster0-minhkiet.bebh5pw.mongodb.net/?retryWrites=true&w=majority")
database: Database = mongo_client.get_database("test")
collection = Collection = database.get_collection("products")
app = Flask(__name__)
CORS(app)


# Read image features
prods = []
products= list(collection.find())
pre = FeatureExtractorManager()
pre.extract_features(products)
fe = FeatureExtractor()

features = []
img_paths = []

for feature_path in Path("./static/feature").glob("*.npy"):
    features.append(np.load(feature_path))
    n_path = feature_path.stem
# Cắt chuỗi
    parts = n_path.split(".")
# Lấy phần tử thứ hai
    string2 = parts[-1]
    img_paths.append(string2)
# print("khoi dau", features)
features = np.array(features)
img_paths = np.array(img_paths)

@app.route('/')
def hello():
    return 'hello'
@app.route('/image', methods=['POST'])
def index():
    if request.method == 'POST':
        try:
            products= []
            file = requests.get(request.json['query_img'])
            print(request)           
            img = Image.open(BytesIO(file.content))
            uploaded_img_path = "static/uploaded/" + datetime.now().isoformat().replace(":", ".") +"find.jpg"
            img = img.convert('RGB')
            img.save(uploaded_img_path)
            query = fe.extract(img)
            knn = NearestNeighbors(n_neighbors=30, metric='euclidean')
            # dists = np.linalg.norm(features-query, axis=1)  # L2 distances to features
            knn.fit(features)

            # Tìm kiếm hàng xóm gần nhất với hình ảnh truy vấn bằng kNN
            distances, indices = knn.kneighbors(query.reshape(1, -1))

            # Lấy các đường dẫn tương ứng với các hàng xóm
            products = img_paths[indices[0]]

            # Loại bỏ các phần tử trùng lặp và chỉ giữ lại vị trí đầu tiên
            unique_products = []
            seen = set()
            for product in products:
                if product not in seen:
                    unique_products.append(product)
                    seen.add(product)

            # Cập nhật danh sách sản phẩm
            products = unique_products

            # Trả về danh sách các sản phẩm tìm thấy
            response = {"data": products, "status": "OK"}
            return jsonify(response)
        except  Exception:
             return jsonify("error")
   
@app.route('/create', methods=['POST'])
def add_image():
    if request.method == 'POST':
        try:
            global features
            global img_paths
            data = request.json
            product_id = data['id']
            image_url = data['images']
            j=0
            new_features = []
            new_img_paths = []

            for im in image_url:
                j=j+1
                file = requests.get(im)
                img = Image.open(BytesIO(file.content))
                img = img.convert('RGB')
                feature = fe.extract(img)

                print ("thong so", feature)
                feature_path = Path("./static/feature") / (str(j)+'.'+str(product_id) + ".npy")  # e.g., ./static/feature/xxx.npy
                np.save(feature_path, feature)
                # features.append(feature)
                # img_paths.append(product_id)
                new_features.append(feature)
                new_img_paths.append(product_id)

            new_features_array = np.array(new_features)
            features = np.concatenate((features, new_features_array))

            # Chắc chắn rằng new_img_paths là một list
            new_img_paths = np.array(new_img_paths)
            img_paths = np.concatenate((img_paths, new_img_paths))


            response = {"status": "OK", "message": "Image added and model updated successfully."}
            return jsonify(response)
        except Exception as e:
            print(e)
            return jsonify({"status": "ERROR", "message": str(e)})
if __name__=="__main__":
   app.run(host='0.0.0.0', port=5000)
    