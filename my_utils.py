import tensorflow as tf

def convert_outputs_to_tensors(dtype):
    def inner(func):
        def tf_func(*args, **kwargs):
            outputs = func(*args, **kwargs)
            return (tf.convert_to_tensor(elem, dtype=dtype) for elem in output)
        return tf_func
    return inner