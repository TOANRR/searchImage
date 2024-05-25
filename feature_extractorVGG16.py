from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
import numpy as np

# See https://keras.io/api/applications/ for details

class FeatureExtractorVGG16:
    def __init__(self):
        base_model = VGG16(weights='imagenet')
        self.model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)

    def extract(self, img):
        """
        Extract a deep feature from an input image
        Args:
            img: from PIL.Image.open(path) or tensorflow.keras.preprocessing.image.load_img(path)

        Returns:
            feature (np.ndarray): deep feature with the shape=(4096, )
        """
        img = img.resize((224, 224))  # VGG must take a 224x224 img as an input
        # print("img", img)
        img = img.convert('RGB')  # Make sure img is color
        # print("img convert", img)
        x = image.img_to_array(img)  # To np.array. Height x Width x Channel. dtype=float32
        # print("x to array", x)
        x = np.expand_dims(x, axis=0)  # (H, W, C)->(1, H, W, C), where the first elem is the number of img
        # print("x sau mo rong", x)
        x = preprocess_input(x)  # Subtracting avg values for each pixel
      #  print("x sau chinh", x)
        feature = self.model.predict(x)[0]  # (1, 4096) -> (4096, )
        # print("model ",  self.model.predict(x))
        # print("feature", feature)
        # print("normal", np.linalg.norm(feature))
        # print("normalize", feature/np.linalg.norm(feature))
        return feature / np.linalg.norm(feature)  # Normalize