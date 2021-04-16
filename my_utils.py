import tensorflow as tf
from functools import wraps

def convert_outputs_to_tensors(dtype):
    def inner(func):
        @wraps(func)
        def tf_func(*args, **kwargs):
            outputs = func(*args, **kwargs)
            return tuple((tf.convert_to_tensor(elem, dtype=dtype) for elem in outputs))
        return tf_func
    return inner