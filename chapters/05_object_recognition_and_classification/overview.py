import tensorflow as tf
import numpy as np

sess = tf.InteractiveSession()

image_batch = tf.constant([
        [  # First Image
            [[0, 255, 0], [0, 255, 0], [0, 255, 0]],
            [[0, 255, 0], [0, 255, 0], [0, 255, 0]]
        ],
        [  # Second Image
            [[0, 0, 255], [0, 0, 255], [0, 0, 255]],
            [[0, 0, 255], [0, 0, 255], [0, 0, 255]]
        ]
    ])
image_batch.get_shape()

print sess.run(image_batch)[0][0][0]

sess.close()

