import tensorflow as tf
from functools import wraps
from transformers import LongformerTokenizer

def convert_outputs_to_tensors(dtype):
    def inner(func):
        @wraps(func)
        def tf_func(*args, **kwargs):
            outputs = func(*args, **kwargs)
            return tuple((tf.convert_to_tensor(elem, dtype=dtype) for elem in outputs))
        return tf_func
    return inner

def get_tokenizer(max_pos):
    tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
    vocab_file, merges_file = tokenizer.save_vocabulary('.')
    tokenizer = LongformerTokenizer(vocab_file, merges_file, tokenizer.init_kwargs.update({'model_max_length' : max_pos}))
    with open('special_tokens.txt') as f:
        sp_tokens = list(filter(None, f.read().split('\n')))
    tokenizer.add_tokens(sp_tokens)
    user_token_indices = [tokenizer.encode('[USER0]')[1:-1][0], tokenizer.encode('[USER1]')[1:-1][0]]
    return tokenizer, user_token_indices