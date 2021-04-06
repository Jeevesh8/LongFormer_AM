import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras.backend as K
from tensorflow_addons.text.crf import crf_log_likelihood
import numpy as np

def get_transition(num_classes, max_val=0.1, min_val=-0.1, is_padded=False):
    '''
    0:None, 1:B-Claim, 2:I-Claim, 3:B-Premise, 4:I-Premise, 5:PAD (optional)
    Creates transition matrix to initialize crf
    '''
    T = np.random.uniform(min_val, max_val, size=(num_classes, num_classes))
    T[0, [2, 4]] = [-10000., -10000.]
    T[1, [0, 1, 3, 4]] = T[3, [0, 1, 2, 3]] = [-10000., -10000., -10000., -10000.]
    T[2, [4]] = T[4, [2]] = [-10000.]
    if is_padded:
        T[5, :] = [-1e-5 for _ in range(num_classes)]
        T[[1, 3], 5] = [-10000., -10000.]
    return T.astype('float32')

def compute_dsc_loss(y_pred, y_true, alpha=0.6):
    y_pred = K.reshape(K.softmax(y_pred), (-1, y_pred.shape[2]))
    y = K.expand_dims(K.flatten(y_true), axis=1)
    probs = tf.gather_nd(y_pred, y, batch_dims=1)
    pos = K.pow(1 - probs, alpha) * probs
    dsc_loss = 1 - (2 * pos + 1) / (pos + 2)
    return dsc_loss

class CRF(tf.keras.layers.Layer):
    def __init__(self, transition_matrix, **kwargs):
        super(CRF, self).__init__(**kwargs)
        self.transitions = tf.Variable(transition_matrix)
    def call(self, inputs, mask=None, training=None):
        if mask is None:
            raw_input_shape = tf.slice(tf.shape(inputs), [0], [2])
            mask = tf.ones(raw_input_shape)
        sequence_lengths = K.sum(K.cast(mask, 'int32'), axis=-1)

        viterbi_sequence, score = tfa.text.crf_decode(
            inputs, self.transitions, sequence_lengths
        )
        if training:
            return viterbi_sequence, inputs, sequence_lengths, self.transitions
        else:
            return viterbi_sequence, sequence_lengths

class TaskModel(tf.keras.models.Model):
    def __init__(self, encoder, 
                 max_trans=0.1, 
                 min_trans=-0.1, 
                 is_padded=True,
                 use_features=False,
                 feature_size=67,
                 use_gru=False, 
                 use_dsc=False, 
                 alpha=0.5, lr=1e-5):
        super(TaskModel, self).__init__()
        self.encoder = encoder
        self.use_features = use_features
        if use_features:
            self.marker_embedding = tf.keras.layers.Embedding(feature_size, 32)
        if is_padded:
            num_classes = 6
        else:
            num_classes = 5
        self.ff = tf.keras.layers.Dense(num_classes)
        self.use_dsc = use_dsc
        self.use_gru = use_gru
        self.alpha = alpha
        if use_gru:
            self.gru = tf.keras.layers.GRU(num_classes, return_sequences=True)
        self.crf_layer = CRF(get_transition(num_classes, max_val=max_trans, min_val=min_trans, is_padded=is_padded))

    def call(self, inputs, features=None, training=True):
        encoded_seq = self.encoder(inputs, training=training)['last_hidden_state']
        if self.use_features:
            embedded_features = self.marker_embedding(features)
            encoded_seq = tf.keras.layers.Concatenate(axis=-1)([encoded_seq, embedded_features])
        if self.use_gru:
            logits = self.gru(encoded_seq)
        else:
            logits = self.ff(encoded_seq)
        crf_predictions = self.crf_layer(logits, mask=inputs['attention_mask'], training=training)
        if training:
            viterbi_sequence, potentials, sequence_length, chain_kernel = crf_predictions
            return viterbi_sequence, potentials, sequence_length, chain_kernel, logits
        else:
            return crf_predictions

    def compute_loss(self, x, y, sample_weight=None, training=True):
        if self.use_features:
            x1, x2 = x
        else:
            x1, x2 = x, None
        viterbi_sequence, potentials, sequence_length, chain_kernel, logits = self(x1, features=x2, training=training)
        crf_loss = -crf_log_likelihood(potentials, y, sequence_length, chain_kernel)[0]
        if self.use_dsc:
            ds_loss = compute_dsc_loss(potentials, y, self.alpha)
            return tf.reduce_mean(crf_loss), tf.reduce_mean(ds_loss)
        else:
            cc_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(y, logits)
            if sample_weight is not None:
                cc_loss = tf.reduce_sum(cc_loss*sample_weight)/tf.reduce_sum(sample_weight)
            return tf.reduce_mean(crf_loss), cc_loss

    def compute_batch_sample_weight(self, labels):
        _, _, counts = tf.unique_with_counts(tf.reshape(labels, [-1]))
        counts = tf.cast(counts, dtype=tf.float32) + tf.keras.backend.epsilon()
        class_weights = tf.math.log(tf.reduce_sum(counts)/counts)
        non_pad = tf.cast(tf.math.not_equal(labels, 5), dtype=tf.float32)
        weighted_labels = tf.gather(class_weights, labels)
        return non_pad*weighted_labels

    def infer_step(self, x):
        if self.use_features:
            x1, x2 = x
        else:
            x1, x2 = x, None
        return self(x1, features=x2, training=False)
