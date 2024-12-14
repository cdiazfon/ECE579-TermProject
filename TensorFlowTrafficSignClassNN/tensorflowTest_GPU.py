import tensorflow as tf
import time

# Define a large matrix multiplication operation
a = tf.random.normal([10000, 10000])
b = tf.random.normal([10000, 10000])

# Time the operation on the GPU
with tf.device('/GPU:0'):
    start = time.time()
    c = tf.matmul(a, b)
    print("GPU computation time:", time.time() - start)

# Time the operation on the CPU for comparison
with tf.device('/CPU:0'):
    start = time.time()
    c = tf.matmul(a, b)
    print("CPU computation time:", time.time() - start)
