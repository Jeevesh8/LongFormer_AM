import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras.backend as K
from tensorflow_addons.text.crf import crf_log_likelihood
import numpy as np

from configs import config
from tree_crf import tree_crf

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
        self.tree_crf = tree_crf()
        self.linear_tree_crf = tf.keras.layers.Dense(len(config['relations_map'])*config['longformer_dim'], use_bias=False)

    def call(self, inputs, training=True):
        encoded_seq = self.encoder(inputs, training=training)['last_hidden_state']
        logits = self.gru(encoded_seq) if self.use_gru else self.ff(encoded_seq)
        crf_predictions = self.crf_layer(logits, mask=inputs['attention_mask'], training=training)
        
        if not training:
            return tuple((*crf_predictions, encoded_seq))
        
        return tuple((*crf_predictions, logits, encoded_seq))
    
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
    
    def format_log_energies(self, log_energies: tf.Tensor, n_comps: tf.Tensor) -> tf.Tensor:
        """
        Args:
            log_energies:   The log energies predicted for different relations by taking dot products between (possibly transformed)embeddings of 
                            argumentative components of each thread; where all threads in the batch were padded with 0's to have same number of components.
                            See self.get_log_energies() for detailed computation. Tensor of shape [batch_size, different relations, M, M]
        
            n_comps:        The actual number of components in each thread the batch.
        Returns:
            The log_energies in the format specified in the comment in tree_crf.py of shape [batch_size, M, M, different relations]
        """
        batch_size, n_rel_types, M, _ = tf.shape(log_energies).numpy()
        
        #No link from dummy component(0)
        log_energies[:, :, 0, :] = tf.constant(-np.inf)
        
        #No link from component to itself
        log_energies = tf.linalg.set_diag(log_energies, diagonal=tf.fill((batch_size, n_rel_types, M), -np.inf))
        
        #Only None type link when referring to no component(0)
        log_energies[:, 1:, :, 0] = tf.constant(-np.inf)
        
        #No link from non-existent components
        log_energies = tf.transpose(log_energies, perm=[1,3,0,2])                                                                                       #[different relations, M(link to), batch_size, M(link from)]
        log_energies = tf.where(tf.range(M)<n_comps+1, log_energies, -np.inf)

        #No link to non-existent components
        log_energies = tf.transpose(log_energies, perm=[0,3,2,1])                                                                                       #[different relations, M(link from), batch_size, M(link to)]
        log_energies = tf.where(tf.range(M)<n_comps+1, log_energies, -np.inf)
        
        #Making shape compatible with tree_crf implementation
        log_energies = tf.transpose(log_energies, perm=[2,1,3,0])                                                                                       #[batch_size, M(link from), M(link to), different relations]
        
        return log_energies
    
    def get_log_energies(self, comp_type_labels: tf.Tensor, encoded_seq: tf.Tensor) -> tf.Tensor:
        """
        Args:
            comp_type_labels:   The actual per-token component type labels of the samples in the batch [batch_size, seq_len]
            encoded_seq:        The longformer encodings of all the tokens in all the threads in the batch [batch_size, seq_len, embedding_size]
        Returns:
            log_energies tensor of shape [ batch_size, M, M, len(config['relations_map']) ] where the (a,b,c,d)-th entry is the log energy of having 
            a link from component number b to component number c of type d, in sample a. For more info on the format see the comment in tree_crf.py.
        """
        begin_comps = tf.cast(tf.logical_or(comp_type_labels==config['arg_components']['B-C'], comp_type_labels==config['arg_components']['B-P']), dtype=tf.uint32)
        n_comps = tf.reduce_sum(begin_comps, axis=-1)
        
        batch_size, max_seq_len = tf.shape(comp_type_labels).numpy()
        M = tf.reduce_max(n_comps)+1

        from_embds = tf.ragged.boolean_mask(encoded_seq, tf.cast(begin_comps, dtype=tf.bool)).to_tensor(default_value=0.)                               #[batch_size, M-1, config['longoformer_dim']]
        from_embds = tf.pad(from_embds, paddings=[[0,0],[1,0],[0,0]], constant_values=1.)                                                               #[batch_size, M, config['longoformer_dim']]
        
        to_embds = tf.reshape( self.linear_tree_crf(from_embds), (batch_size, max_seq_len, len(config['relations_map']), config['longformer_dim']))
        to_embds = tf.transpose(to_embds, perm=[0,2,1,3])                                                                                               #[batch_size, different relations, M, config['longformer_dim']]
        
        log_energies = tf.matmul(from_embds, to_embds, transpose_b=True)
        
        return self.format_log_energies(log_energies, n_comps)

    def compute_loss(self, x, y):
        comp_type_labels, refers_to_and_type = y
        viterbi_sequence, potentials, sequence_length, chain_kernel, logits, encoded_seq = self.call(x, training=True)
        crf_loss = -crf_log_likelihood(potentials, comp_type_labels, sequence_length, chain_kernel)[0]
        comp_type_cc_loss = self.get_cross_entropy(logits, comp_type_labels, x['attention_mask'], len(config['arg_components']))
        log_energies = self.get_log_enegies(comp_type_labels, encoded_seq)
        relations_loss = self.tree_crf.disc_loss(log_energies, refers_to_and_type)
        return tf.reduce_mean(crf_loss), comp_type_cc_loss, relations_loss
    
    def infer_step(self, x):
        viterbi_seqs, seq_lens, encoded_seq = self(x, training=False)
        log_energies = self.get_log_enegies(viterbi_seqs, encoded_seq)
        _, optimal_trees = self.tree_crf.mst(log_energies)
        return viterbi_seqs, seq_lens, optimal_trees