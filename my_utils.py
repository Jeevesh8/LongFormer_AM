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

def _create_min_max_boundaries(max_length,
                               min_boundary=256,
                               boundary_scale=1.1):
  """Create min and max boundary lists up to max_length.
  For example, when max_length=24, min_boundary=4 and boundary_scale=2, the
  returned values will be:
    buckets_min = [0, 4, 8, 16, 24]
    buckets_max = [4, 8, 16, 24, 25]
  Args:
    max_length: The maximum length of example in dataset.
    min_boundary: Minimum length in boundary.
    boundary_scale: Amount to scale consecutive boundaries in the list.
  Returns:
    min and max boundary lists
  """
  # Create bucket boundaries list by scaling the previous boundary or adding 1
  # (to ensure increasing boundary sizes).
  bucket_boundaries = []
  x = min_boundary
  while x < max_length:
    bucket_boundaries.append(x)
    x = max(x + 1, int(x * boundary_scale))

  # Create min and max boundary lists from the initial list.
  buckets_min = [0] + bucket_boundaries
  buckets_max = bucket_boundaries + [max_length + 1]
  return buckets_min, buckets_max