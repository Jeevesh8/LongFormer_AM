import tensorflow as tf
from functools import wraps
from transformers import LongformerTokenizer, TFLongformerForMaskedLM, LongformerForMaskedLM

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

def get_model(max_pos, attention_window):
    model = LongformerForMaskedLM.from_pretrained('allenai/longformer-base-4096', attention_window=attention_window)
    model_config = model.config
    
    current_max_pos, embed_size = model.longformer.embeddings.position_embeddings.weight.shape
    model_config.max_position_embeddings = max_pos
    new_pos_embed = model.longformer.embeddings.position_embeddings.weight.new_empty(max_pos, embed_size)

    # copy position embeddings over and over to initialize the new position embeddings
    k = 2
    step = current_max_pos-2
    while k<max_pos-1:
        new_pos_embed[k:(k + step)] = model.longformer.embeddings.position_embeddings.weight[2:]
        k += step

    model.longformer.embeddings.position_embeddings.weight.data = new_pos_embed
    model.longformer.embeddings.position_ids.data = torch.tensor([i for i in range(max_pos)]).reshape(1, max_pos)
    model.save_pretrained('tmp/LF/')

    model = TFLongformerForMaskedLM.from_pretrained('tmp/LF', from_pt=True, attention_window=attention_window)
    return model