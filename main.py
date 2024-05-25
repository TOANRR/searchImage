import numpy as np
from PIL import Image
import pillow_avif
from feature_extractor import FeatureExtractor
from feature_extractorVGG16 import FeatureExtractorVGG16
from feature_extractorResnet50V2 import FeatureExtractorResnet50V2

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
import os
import glob

mongo_client: MongoClient = MongoClient("mongodb+srv://minhkietpeople:TNJ99tPbdFhNIbOH@cluster0-minhkiet.bebh5pw.mongodb.net/?retryWrites=true&w=majority")
database: Database = mongo_client.get_database("test")
collection = Collection = database.get_collection("products")
app = Flask(__name__)
CORS(app)


# Read image features
# prods = []
# products= list(collection.find())
# pre = FeatureExtractorManager()
# pre.extract_features(products)


fe = FeatureExtractorResnet50V2()

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
            knn = NearestNeighbors(n_neighbors=100, metric='euclidean')
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
            # print(products)
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
        

@app.route('/delete/<product_id>', methods=['DELETE'])
def delete_product(product_id):
    try:
        global features
        global img_paths

        # Tìm vị trí của các mục cần xóa trong mảng numpy feature
        indices_to_delete = [i for i, path in enumerate(img_paths) if product_id in path]
        print(indices_to_delete)


        if not indices_to_delete:
            return jsonify({"status": "ERROR", "message": "No features to delete for the given product ID."}), 404

        # Xóa các tệp tương ứng trong thư mục static/feature

        # Đường dẫn tới thư mục chứa các tệp .npy
        directory = "./static/feature/"

        # Tìm tất cả các tệp có dạng *.product_id.npy
        files_to_delete = glob.glob(f"{directory}*.{product_id}.npy")

        # Xóa từng tệp trong danh sách tìm được
        for file_path in files_to_delete:
            os.remove(file_path)

        # Xóa các mục tương ứng trong mảng numpy feature
        features = np.delete(features, indices_to_delete, axis=0)
        img_paths = np.delete(img_paths, indices_to_delete)

        # Trả về phản hồi
        return jsonify({"status": "OK", "message": "Features deleted successfully."}), 200

    except Exception as e:
        return jsonify({"status": "ERROR", "message": str(e)}), 500


@app.route('/update', methods=['POST'])
def update_product_features():
    try:
        # Nhận dữ liệu từ yêu cầu POST
        data = request.json

        # Lấy ID sản phẩm và URL hình ảnh mới từ dữ liệu nhận được
        product_id = data.get('id')
        new_image_urls = data.get('images')

        # Kiểm tra xem có ID sản phẩm và URL hình ảnh mới không
        if product_id is None or new_image_urls is None:
            return jsonify({'status': 'ERR', 'message': 'Missing product ID or new image URLs'}), 400

        # Lấy các vị trí của các mục cần cập nhật trong mảng img_paths
        indices_to_update = [i for i, path in enumerate(img_paths) if product_id in path]

        # Nếu không tìm thấy mục nào cần cập nhật, trả về thông báo lỗi
        if not indices_to_update:
            return jsonify({'status': 'ERR', 'message': 'No features to update for the given product ID'}), 404

        # Xóa các tệp tương ứng với sản phẩm trong thư mục static/feature
        directory = "./static/feature/"
        files_to_delete = [f"{directory}{idx}.{product_id}.npy" for idx in range(1, len(new_image_urls) + 1)]
        for file_path in files_to_delete:
            os.remove(file_path)

        # Cập nhật feature cho hình ảnh mới
        new_features = []
        for idx, image_url in enumerate(new_image_urls, start=1):
            file = requests.get(image_url)
            img = Image.open(BytesIO(file.content))
            img = img.convert('RGB')
            feature = fe.extract(img)
            new_features.append(feature)

# Sau khi đã xử lý hết tất cả các hình ảnh, lưu toàn bộ danh sách new_features vào các tệp NPY
        for idx, feature in enumerate(new_features, start=1):
            np.save(f"{directory}{idx}.{product_id}.npy", feature)

        # Cập nhật features tương ứng với các vị trí cần cập nhật
        new_features_array = np.array(new_features)
        for idx in indices_to_update:
            features[idx] = new_features_array[idx % len(new_features)]

        return jsonify({'status': 'OK', 'message': 'Product features updated successfully'}), 200

    except Exception as e:
        print(e)
        idx =1
        for id in indices_to_update:
             np.save(f"{directory}{idx}.{product_id}.npy", features[id])
             idx= idx+1
        return jsonify({'status': 'ERR', 'message': str(e)}), 200
if __name__=="__main__":
   app.run(host='0.0.0.0', port=5000)
    