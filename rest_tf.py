import json
import os
import numpy as np
from PIL import Image
import requests
import pdb


def single_predict(IMAGE_EXAMPLE_PATH):
    # IMAGE_EXAMPLE_PATH = "handwritten_7.png"
    MODEL_SIGNATURE_NAME = "serving_default"

    image = Image.open(IMAGE_EXAMPLE_PATH)
    image = np.expand_dims(np.array(image), axis=0)  # image.shape: (1, 28, 28)
    # pdb.set_trace()
    image2 = image.tolist()
    req = json.dumps({"signature_name": MODEL_SIGNATURE_NAME, "instances": image2})  # type(req): <class 'str'>

    IMAGE_EXAMPLE_CLASS = 7
    MODEL_NAME = "my_model"
    PORT = 8501
    SERVER = f"http://localhost:{PORT}/v1/models/{MODEL_NAME}:predict"

    res = requests.post(SERVER, data=req)
    res.raise_for_status()
    probs = res.json()['predictions']
    pred = np.argmax(probs, axis=1)[0]
    print(
        f"image: {IMAGE_EXAMPLE_PATH}, "
        f"Predicted category: {pred}, "
        f"actual: {IMAGE_EXAMPLE_CLASS}")

path1 = 'mnist_png/mnist_png/testing/7/'
list1 = os.listdir(path1)
for i in list1:
    file1 = path1 + i
    single_predict(file1)

pdb.set_trace()
print('done')
