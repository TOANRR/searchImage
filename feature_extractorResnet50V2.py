from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.applications.resnet_v2 import preprocess_input

from tensorflow.keras.models import Model
import numpy as np

class FeatureExtractorResnet50V2:
    def __init__(self):
        self.model= ResNet50V2(
        include_top=True,
        weights='imagenet',
        input_tensor=None,
        input_shape=(224, 224, 3),
        pooling=None,
        classes=1000,
        classifier_activation='softmax'
)

    def extract(self, img):
        """
        Extract a deep feature from an input image
        Args:
            img: from PIL.Image.open(path) or tensorflow.keras.preprocessing.image.load_img(path)

        Returns:
            feature (np.ndarray): deep feature with the shape=(2048, )
        """
        img = img.resize((224, 224))  # ResNet50 must take a 224x224 img as an input
        img = img.convert('RGB')  # Make sure img is color
        x = image.img_to_array(img)  # To np.array. Height x Width x Channel. dtype=float32
        x = np.expand_dims(x, axis=0)  # (H, W, C)->(1, H, W, C), where the first elem is the number of img
        x = preprocess_input(x)  # Subtracting avg values for each pixel
        feature = self.model.predict(x)[0]  # (1, 2048) -> (2048, )
        return feature / np.linalg.norm(feature)  # Normalize

# Example usage:
# img = Image.open('path_to_image.jpg')
# fe = FeatureExtractor()
# feature = fe.extract(img)
# print(feature)
