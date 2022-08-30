import grpc
import numpy as np
import os
from PIL import Image
import tensorflow as tf
from tensorflow_serving.apis import prediction_service_pb2_grpc
from tensorflow_serving.apis import predict_pb2
import pdb


def single_predict(IMAGE_EXAMPLE_PATH):
    PORT = 8500
    SERVER = f"localhost:{PORT}"

    channel = grpc.insecure_channel(SERVER)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel=channel)

    # IMAGE_EXAMPLE_PATH = "handwritten_7.png"
    IMAGE_EXAMPLE_CLASS = 7
    MODEL_NAME = "my_model"
    INPUT_NAME = "image_data"
    MODEL_SIGNATURE_NAME = "serving_default"

    image = Image.open(IMAGE_EXAMPLE_PATH)            # image.size: (28, 28)     np.array(image).shape: (28, 28)
    image = np.expand_dims(np.array(image), axis=0)   # image.size: 784   image.shape: (1, 28, 28)

    req = predict_pb2.PredictRequest()
    req.model_spec.name = MODEL_NAME
    req.model_spec.signature_name = MODEL_SIGNATURE_NAME
    req.inputs[INPUT_NAME].CopyFrom(tf.make_tensor_proto(image, shape=image.shape, dtype=tf.float32))

    OUTPUT_NAME = "probabilities"
    res = stub.Predict(req, 10.0)
    probs = tf.make_ndarray(res.outputs[OUTPUT_NAME])
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
