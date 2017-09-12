import os

import tensorflow as tf
import numpy as np
import matplotlib as mil

sess = tf.InteractiveSession()
mil.use('svg')

from matplotlib import pyplot
from matplotlib import backends

# mil.use("nbagg")
fig = pyplot.gcf()
fig.set_size_inches(4, 4)

image_filename = [os.path.dirname(__file__) + "/" + "n02113023_219.jpg"]
# image_filename = "/Users/erikerwitt/Downloads/images/n02085936-Maltese_dog/n02085936_804.jpg"

# filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once(image_filename))

filename_queue=tf.train.string_input_producer(image_filename)

image_reader = tf.WholeFileReader()
_, image_file = image_reader.read(filename_queue)
image = tf.image.decode_jpeg(image_file)

sess.run(tf.global_variables_initializer())

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord=coord)

image_batch = tf.image.convert_image_dtype(tf.expand_dims(image, 0), tf.float32, saturate=False)

kernel = tf.constant([
    [
        [[-1., 0., 0.], [0., -1., 0.], [0., 0., -1.]],
        [[-1., 0., 0.], [0., -1., 0.], [0., 0., -1.]],
        [[-1., 0., 0.], [0., -1., 0.], [0., 0., -1.]]
    ],
    [
        [[-1., 0., 0.], [0., -1., 0.], [0., 0., -1.]],
        [[8., 0., 0.], [0., 8., 0.], [0., 0., 8.]],
        [[-1., 0., 0.], [0., -1., 0.], [0., 0., -1.]]
    ],
    [
        [[-1., 0., 0.], [0., -1., 0.], [0., 0., -1.]],
        [[-1., 0., 0.], [0., -1., 0.], [0., 0., -1.]],
        [[-1., 0., 0.], [0., -1., 0.], [0., 0., -1.]]
    ]
])

conv2d = tf.nn.conv2d(image_batch, kernel, [1, 1, 1, 1], padding="SAME")
activation_map = sess.run(tf.minimum(tf.nn.relu(conv2d), 255))

fig = pyplot.gcf()
pyplot.imshow(activation_map[0], interpolation='nearest')
# pyplot.show()
fig.set_size_inches(4, 4)
fig.savefig("./images/chapter-05-object-recognition-and-classification/convolution/example-edge-detection.png")
