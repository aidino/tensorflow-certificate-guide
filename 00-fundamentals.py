import tensorflow as tf
import numpy as np

def function(x, y):
    return x**2 + y

@tf.function
def tf_function(x, y):
    return x ** 2 + y

if __name__ == '__main__':
    print(f"function(2, 2): {function(2, 2)}")
    print(f"tf_function(2, 2): {tf_function(2, 2)}")