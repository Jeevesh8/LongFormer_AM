import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras.backend as K
from tensorflow_addons.text.crf import crf_log_likelihood
import numpy as np

from configs import config

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
        T[5, 5] = 0
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
        
        return viterbi_sequence, sequence_lengths

class TaskModel(tf.keras.models.Model):
    def __init__(self, encoder, 
                 max_trans=0.1, 
                 min_trans=-0.1, 
                 is_padded=False,
                 use_gru=False, 
                 alpha=0.5, lr=1e-5):
        super(TaskModel, self).__init__()
        self.encoder = encoder
        num_classes = len(config['arg_components']) #+1
        self.ff = tf.keras.layers.Dense(num_classes)
        self.use_gru = use_gru
        self.alpha = alpha
        if use_gru:
            self.gru = tf.keras.layers.GRU(num_classes, return_sequences=True)
        self.crf_layer = CRF(get_transition(num_classes, max_val=max_trans, min_val=min_trans, is_padded=is_padded))
        self.relation_type_predictor = tf.keras.layers.Dense(len(config['relations']))
        self.refers_predictor = tf.keras.layers.Dense(config['max_rel_length'])
        
    def call(self, inputs, training=True):
        encoded_seq = self.encoder(inputs, training=training)['last_hidden_state']
        logits = self.gru(encoded_seq) if self.use_gru else self.ff(encoded_seq)
        crf_predictions = self.crf_layer(logits, mask=inputs['attention_mask'], training=training)
        
        if not training:
            return tuple((*crf_predictions, self.relation_type_predictor(encoded_seq), self.refers_predictor(encoded_seq)))
        
        return tuple((*crf_predictions, logits, self.relation_type_predictor(encoded_seq), self.refers_predictor(encoded_seq)))
    
    def compute_batch_sample_weight(self, labels, pad_mask, max_possible_length):
        counts = tf.reduce_sum(tf.cast(tf.equal(tf.expand_dims(tf.range(0, max_possible_length), -1), tf.reshape(labels, [-1])), dtype=tf.int32), axis=-1)
        counts = tf.cast(counts, dtype=tf.float32) + tf.keras.backend.epsilon()
        class_weights = tf.math.log(tf.reduce_sum(counts)/counts)
        non_pad = tf.cast(pad_mask, dtype=tf.float32)
        weighted_labels = tf.gather(class_weights, labels)
        return non_pad*weighted_labels

    def get_cross_entropy(self, logits, labels, pad_mask, max_possible_length):
        sample_weight = self.compute_batch_sample_weight(labels, pad_mask, max_possible_length=max_possible_length)
        cc_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels, logits)
        n_samples = tf.reduce_sum(sample_weight)
        return tf.reduce_sum(cc_loss*sample_weight)/n_samples if n_samples!=0 else tf.convert_to_tensor(0.)
    
    def compute_loss(self, x, y):
        comp_type_labels, relation_type_labels, refers_labels = y
        viterbi_sequence, potentials, sequence_length, chain_kernel, logits, relation_type_logits, refers_logits = self.call(x, training=True)
        crf_loss = -crf_log_likelihood(potentials, comp_type_labels, sequence_length, chain_kernel)[0]
        comp_type_cc_loss = self.get_cross_entropy(logits, comp_type_labels, x['attention_mask'], len(config['arg_components']))
        relation_type_cc_loss = self.get_cross_entropy(relation_type_logits, relation_type_labels, x['attention_mask'], len(config['relations']))
        refers_cc_losses = tf.map_fn(lambda labels: self.get_cross_entropy(refers_logits, labels, x['attention_mask'], config['max_rel_length']), tf.transpose(refers_labels, perm=(2,0,1)), fn_output_signature=tf.TensorSpec(shape=[], dtype=tf.float32))
        refers_cc_loss = tf.reduce_sum(refers_cc_losses)
        return tf.reduce_mean(crf_loss), comp_type_cc_loss, relation_type_cc_loss, refers_cc_loss
    
    def infer_step(self, x):
        viterbi_seqs, seq_lens, relation_type_logits, refers_logits = self(x, training=False)
        relation_type_preds = tf.argmax(relation_type_logits, axis=-1)
        refers_preds = tf.argmax(refers_logits, axis=-1)
        return viterbi_seqs, seq_lens, relation_type_preds, refers_preds