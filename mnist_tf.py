# import tensorflow.compat.v1 as tf
import tensorflow as tf
from tensorflow.python import ipu
import os
import shutil
import numpy as np
from PIL import Image
import pdb

def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    return model

cfg = ipu.config.IPUConfig()
cfg.auto_select_ipus = 1
cfg.configure_ipu_system()

# bs = 32
# epochs = 4
# ckpt_path = f'ckpts/model.ckpt'
# graph = tf.Graph()
# with graph.as_default():
#     mnist = tf.keras.datasets.mnist
#     (x_train, y_train), tmp = mnist.load_data()
#     x_train = x_train / 255.0
#     train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(bs, drop_remainder=True)
#     train_ds = train_ds.map(lambda d, l: (tf.cast(d, tf.float32), tf.cast(l, tf.int32)))
#     train_ds = train_ds.repeat()
#     train_iter = tf.data.make_initializable_iterator(train_ds)
#     (x, y) = train_iter.get_next()

#     model = create_model()
    
#     def training_loop(x, y):
#         logits = model(x, training=True)
#         # loss = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=logits)  #WARNING:tensorflow:From mnist_tf.py:34: The name tf.losses.sparse_softmax_cross_entropy is deprecated. Please use tf.compat.v1.losses.sparse_softmax_cross_entropy instead.
#         # opti = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss=loss)  #WARNING:tensorflow:From mnist_tf.py:35: The name tf.train.AdamOptimizer is deprecated. Please use tf.compat.v1.train.AdamOptimizer instead.
#         loss = tf.compat.v1.losses.sparse_softmax_cross_entropy(labels=y, logits=logits)
#         opti = tf.compat.v1.train.AdamOptimizer(learning_rate=0.01).minimize(loss=loss)
#         return (loss, opti)

#     with ipu.scopes.ipu_scope('/device:IPU:0'):
#         ipu_training_loop = ipu.ipu_compiler.compile(computation=training_loop, inputs=[x, y])

#     # saver = tf.train.Saver()
#     # with tf.Session() as sess:
#     #     sess.run(tf.global_variables_initializer())
#     #     sess.run(train_iter.initializer)
#     #     batches_per_epoch = len(x_train) // bs
#     #     for i in range(epochs):
#     #         total_loss = 0.0
#     #         for j in range(batches_per_epoch):
#     #             loss = sess.run(ipu_training_loop)
#     #             total_loss += loss[0]
#     #         print(f'Loss: {total_loss / batches_per_epoch}')
#     #     save_path = saver.save(sess, ckpt_path)
#     #     print('save_path:', save_path)


#-------------------------
ckpt_path = f'ckpts_0830/model.ckpt'

def var_init(sess):
    saver = tf.train.Saver()
    ipu.utils.move_variable_initialization_to_cpu()
    init = tf.global_variables_initializer()
    sess.run(init)
    saver.restore(sess, ckpt_path)

EXPORT_BATCH_SIZE = 1
MODEL_VERSION = 1
MODEL_NAME = "my_model"
SAVED_MODEL_PATH = f"{MODEL_NAME}/{MODEL_VERSION}"
OUTPUT_NAME = "probabilities"

if os.path.exists(SAVED_MODEL_PATH):
    shutil.rmtree(SAVED_MODEL_PATH)

export_graph = tf.Graph()
with export_graph.as_default():
    model = create_model()

    def predict(image_data):
        return model(image_data)

    input_signature = (
        tf.TensorSpec(
            shape=(EXPORT_BATCH_SIZE,28,28),
            dtype=tf.float32),
    )
    # runtime_func = ipu.serving.export_single_step(
    #     predict,
    #     SAVED_MODEL_PATH,
    #     iterations=10,
    #     output_names=[OUTPUT_NAME],
    #     variable_initializer=var_init,
    #     input_signature=input_signature)  #TypeError: export_single_step() got an unexpected keyword argument 'input_signature'
    runtime_func = ipu.serving.export_single_step(
        predict,
        SAVED_MODEL_PATH,
        10,
        input_signature,
        None,
        var_init,
        [OUTPUT_NAME]
    )


IMAGE_EXAMPLE_CLASS = 7
IMAGE_EXAMPLE_PATH = "handwritten_7.png"

image = Image.open(IMAGE_EXAMPLE_PATH)
image = np.expand_dims(np.array(image, dtype=np.float32), axis=0)

with tf.Session() as sess:
    input_placeholder = tf.placeholder(dtype=np.float32, shape=(1,28,28))
    result_op = runtime_func(input_placeholder)
    result = sess.run(result_op, feed_dict={input_placeholder: image})
    probabilities = result[0]
    predicted_category = np.argmax(probabilities, axis=1)[0]
    print(f"Predicted category: {predicted_category}, "
          f"actual: {IMAGE_EXAMPLE_CLASS}")


pdb.set_trace()
print('done')
