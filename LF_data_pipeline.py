PAD_LABEL = 5
B_CLAIM = 1
I_CLAIM = 2
B_PREMISE = 3
I_PREMISE = 4
import random
import utils
import tensorflow as tf
import glob

def load_labeled_data(file_list):
    '''
    Read threads along with claim-premise segments
    '''
    if isinstance(file_list, list):
        file_names = file_list
    elif isinstance(file_list, str):
        file_names = glob.glob(file_list+'/*.txt')
    def example_generator():
        for file_name in random.sample(file_names, len(file_names)):
            idx = file_name[:-4]
            content_file = file_name
            claim_file = idx + '.claim'
            premise_file = idx + '.premise'
            with open(content_file) as f:
                content = f.read()
            with open(claim_file) as f:
                claims = f.read()
            with open(premise_file) as f:
                premises = f.read()
            yield tf.constant(idx), tf.constant(content), tf.constant(claims), tf.constant(premises)

    return tf.data.Dataset.from_generator(example_generator, 
                                          (tf.string, tf.string, tf.string, tf.string),
                                          (tf.TensorShape([]), tf.TensorShape([]), tf.TensorShape([]), tf.TensorShape([]))
                                         )
def load_unlabeled_data(dir_name):
    '''
    Read threads from directory without labels
    '''
    def example_generator():
        for file_name in glob.glob(dir_name+'/*.txt'):
            idx = file_name[:-4].split('/')[-1]
            content_file = file_name
            with open(content_file) as f:
                content = f.read()
            yield tf.constant(idx), tf.constant(content)
    return tf.data.Dataset.from_generator(example_generator, 
                                          (tf.string, tf.string),
                                          (tf.TensorShape([]), tf.TensorShape([]))
                                         )
def clean_text(text):
    pattern0 = '(\n\&gt; \*Hello[\s\S]*)|(\_|\*|\#)+'
    pattern1 = "(https?:\/\/)(\s)*(www\.)?(\s)*((\w|\s)+\.)*([\w\-\s]+\/)*([\w\-]+)((\?)?[\w\s]*=\s*[\w\%&]*)*(\.html|\.htm)*"
    pattern2 = "\&gt;(.*)\n\n"
    text = tf.strings.regex_replace(text, pattern1, '[URL]')
    text = tf.strings.regex_replace(text, pattern0, '')
    text = tf.strings.regex_replace(text, pattern2, '[STARTQ]'+r'\1'+' [ENDQ] ')
    return text

def clean_with_label(idx, content, claim, premise):
    return idx, clean_text(content), clean_text(claim), clean_text(premise)

def clean_wo_label(idx, content):
    return idx, clean_text(content)

def return_tokenize_fn(tokenizer, user_tokens, is_labeled=True, with_marker_feature=False, markers=None):
    def get_global_attention(tokens):
        g_att = tf.math.equal(tokens, tokenizer.encode(user_tokens[0])[1])
        for t in user_tokens[1:]:
            g_att = tf.math.logical_or(g_att, tf.math.equal(tokens, tokenizer.encode(t)[1]))
        return tf.cast(g_att, dtype=tf.int32)
    def tokenize_w_label(idx, content, claim, premise):
        inputs = tokenizer(content.numpy().decode('utf-8'), truncation=True, return_tensors='tf')
        enc_tokens = list(inputs['input_ids'].numpy()[0])
        labels = [0 for _ in enc_tokens]
        for claim in filter(None, claim.numpy().decode('utf-8').split('\n')):
            indices = utils.find_sub_list(tokenizer.encode(claim)[1:-1], enc_tokens)
            if indices is not None:
                labels[indices[0]] = B_CLAIM
                for i in range(indices[0]+1, indices[1]+1):
                    labels[i] = I_CLAIM
        for premise in filter(None, premise.numpy().decode('utf-8').split('\n')):
            indices = utils.find_sub_list(tokenizer.encode(premise)[1:-1], enc_tokens)
            if indices is not None:
                labels[indices[0]] = B_PREMISE
                for i in range(indices[0]+1, indices[1]+1):
                    labels[i] = I_PREMISE
        labels = tf.convert_to_tensor(labels, dtype=tf.int32)
        return (idx, 
                    tf.squeeze(inputs['input_ids']),
                    tf.squeeze(inputs['attention_mask']),
                    tf.squeeze(get_global_attention(inputs['input_ids'])),
                    labels)

    def tokenize_wo_label(idx, content):
        inputs = tokenizer(content.numpy().decode('utf-8'), truncation=True, return_tensors='tf')
        return (idx, 
                tf.squeeze(inputs['input_ids']), 
                tf.squeeze(inputs['attention_mask']), 
                tf.squeeze(get_global_attention(inputs['input_ids'])))
    if is_labeled:
        return tokenize_w_label
    else:
        return tokenize_wo_label


def load_data_w_label(dir_name, tokenizer, user_tokens):
    dataset = load_labeled_data(dir_name)
    dataset = dataset.map(clean_with_label)
    dataset = dataset.map(lambda x,y,z,w:tf.py_function(return_tokenize_fn(tokenizer, user_tokens, is_labeled=True),
                                                  inp=[x,y,z,w],
                                                  Tout=(tf.string, tf.int32, tf.int32, tf.int32, tf.int32)), #, tf.float32)), 
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return dataset

def load_data_wo_label(dir_name, tokenizer, user_tokens):
    dataset = load_unlabeled_data(dir_name)
    dataset = dataset.map(clean_wo_label)
    dataset = dataset.map(lambda x,y:tf.py_function(return_tokenize_fn(tokenizer, user_tokens, is_labeled=False),
                                                  inp=[x,y],
                                                  Tout=(tf.string, tf.int32, tf.int32, tf.int32)), 
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return dataset

def _get_example_length(i):
    return tf.shape(i)[-1]

def _create_min_max_boundaries(max_length,
                               min_boundary=256,
                               boundary_scale=1.1):
    bucket_boundaries = []
    x = min_boundary
    while x < max_length:
        bucket_boundaries.append(x)
        x = max(x + 1, int(x * boundary_scale))

    buckets_min = [0] + bucket_boundaries
    buckets_max = bucket_boundaries + [max_length + 1]
    return buckets_min, buckets_max


def _batch_labeled_examples(dataset, batch_size, max_length, tokenizer):
    buckets_min, buckets_max = _create_min_max_boundaries(max_length)
    bucket_batch_sizes = [int(batch_size) // x for x in buckets_max]
    bucket_batch_sizes = tf.constant(bucket_batch_sizes, dtype=tf.int64)

    def example_to_bucket_id(i0, i1, i2, i3, i4): #, i5):
        seq_length = _get_example_length(i1)

        conditions_c = tf.logical_and(
            tf.less_equal(buckets_min, seq_length), tf.less(seq_length,
                                                        buckets_max))
        bucket_id = tf.reduce_min(tf.where(conditions_c))
        return bucket_id

    def window_size_fn(bucket_id):
        return bucket_batch_sizes[bucket_id]

    def batching_fn(bucket_id, grouped_dataset):
        bucket_batch_size = window_size_fn(bucket_id)

        return grouped_dataset.padded_batch(bucket_batch_size,
                                            padded_shapes=([],[None],[None],[None],[None]), #[None]),
                                            padding_values=(None, 
                                                            tokenizer.pad_token_id,
                                                            0, 0, PAD_LABEL) #, 0.)
                                           )

    return dataset.apply(
        tf.data.experimental.group_by_window(
          key_func=example_to_bucket_id,
          reduce_func=batching_fn,
          window_size=None,
          window_size_func=window_size_fn)).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


def _batch_unlabeled_examples(dataset, batch_size, max_length, tokenizer):
    buckets_min, buckets_max = _create_min_max_boundaries(max_length)
    bucket_batch_sizes = [int(batch_size) // x for x in buckets_max]
    bucket_batch_sizes = tf.constant(bucket_batch_sizes, dtype=tf.int64)

    def example_to_bucket_id(i0, i1, i2, i3):
        seq_length = _get_example_length(i1)

        conditions_c = tf.logical_and(
            tf.less_equal(buckets_min, seq_length), tf.less(seq_length,
                                                        buckets_max))
        bucket_id = tf.reduce_min(tf.where(conditions_c))
        return bucket_id

    def window_size_fn(bucket_id):
        return bucket_batch_sizes[bucket_id]

    def batching_fn(bucket_id, grouped_dataset):
        bucket_batch_size = window_size_fn(bucket_id)

        return grouped_dataset.padded_batch(bucket_batch_size,
                                            padded_shapes=([],[None],[None],[None]),
                                            padding_values=(None, 
                                                            tokenizer.pad_token_id,
                                                            0, 0)
                                           )
    return dataset.apply(
        tf.data.experimental.group_by_window(
          key_func=example_to_bucket_id,
          reduce_func=batching_fn,
          window_size=None,
          window_size_func=window_size_fn)).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

def make_dataset(dir_name, tokenizer,
                 maxlen=4096, 
                 batch_size=2048,
                 is_labeled=True):
    with open('special_tokens.txt') as f:
        user_tokens = list(filter(None, f.read().split('\n')))[3:]
    if is_labeled:
        return _batch_labeled_examples(load_data_w_label(dir_name, tokenizer, user_tokens), 
                                       batch_size, 
                                       maxlen,
                                       tokenizer)
    else:
        return _batch_unlabeled_examples(load_data_wo_label(dir_name, tokenizer, user_tokens), 
                                         batch_size, 
                                         maxlen,
                                         tokenizer)
