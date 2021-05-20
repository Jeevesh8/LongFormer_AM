import tensorflow as tf
from typing import List, Tuple
import numpy as np

from configs import config
from my_utils import convert_tensor_to_lists
"""

-------------------------------------------------A COMMENT ON THE STRUCTURE OF LOG_ENERGIES---------------------------------------------------------------

1. The indices going down, to the extreme left indicate a link from that index, and indices along the row at the top indicate a link to that index.

2. No link from 0-th index to any index can be made.(A link to 0-th index denotes the component refers to no other component, i.e., refers = "None")

3. No link from a component to itself, is possible. So all the dash(-)ed positions in the matrix below, have negative infinity in that location for 
all relation types. 

4. If a component is related to no other component, its relation type is autmatically "None". This means that positions marked with (*) have 
negative infinity in all relation types, except "None".

5. If the below matrix A, is to be padded with another matrix, corresponding to another thread, with say 3 components, then that thread's matrix(Matrix B)
will be extended to size 5 and have all the extra(e) positions, filled with negative infinity, for all relation types.

---------Matrix A--------
    0   1   2   3   4   5
0   -   -   -   -   -   -
1   *   -
2   *       -
3   *           -
4   *               -
5   *                   -

---------Matrix B--------
    0   1   2   3   4   5
0   -   -   -   -   e   e
1   *   -           e   e
2   *       -       e   e
3   *           -   e   e
4   e   e   e   e   e   e
5   e   e   e   e   e   e

---------------------------------------------------------------------------------------------------------------------------------------------------------------

"""

class tree_crf(tf.keras.layers.Layer):
    def __init__(self, prior=None):
        super(tree_crf, self).__init__()
        self.prior = prior

    def flat_idx_to_tuple(self, idx: tf.Tensor, dimensions: List[int]) -> Tuple[tf.Tensor]:
        """Assume a tensor of size (A,B,C...) is flattened to size [-1], this function converts index idx in the tensor of shape [-1],
        to the corresponding idx in the original tensor.
        Args:
            idx:            index in the flattened tensor
            dimensions:     The sizes of various dimensions that were flattened, except the first.
        Returns:
            A tuple of len(dimensions)+1 tensors where the i-th tensor consists of indices along the i-th dimension, which was flattened. 
        """
        dimensions = [int(tf.reduce_max(idx).numpy())+1]+dimensions+[1]
        tuple_indices = [0]
        remaining_indices = idx
        for i in range(len(dimensions)-1, 0, -1):
            remaining_indices = (remaining_indices-tuple_indices[-1])/dimensions[i]
            remaining_indices = tf.cast(remaining_indices, dtype=tf.int32)
            tuple_indices.append( tf.math.floormod( remaining_indices, dimensions[i-1]) )
        tuple_indices.reverse()
        return tuple(tuple_indices[:-1])

    def mst(self, log_energies: tf.Tensor)-> Tuple[tf.Tensor, List[List[Tuple[int,int,int]]]]:
        """
        Args:
            log_energies:   A tensor of size [batch_size, M, M, number of relation types] where M = max{1+number of components in i-th sample of batch}
                            padded with -infinity. (i,j,k)-th entry indicates the log energy of component i referring to component j with relation type k.
        Returns:
            Tensor of size [batch_size] where the i-th entry denotes cost of MST for i-th sample and 
            A list of [list of N_i tuples of form (link_from, link_to, relation_type)] for each sample i in the batch. (where N_i is the number of componenets in the i-th sample of batch)
        """
        energies_shape = tf.shape(log_energies).numpy()
        batch_size, M, n_rel_types = energies_shape[0], energies_shape[1], energies_shape[3]
        partitions = [ [[i] for i in range(M)] for _ in range(batch_size) ]
        mst_energies = [tf.constant(0.) for _ in range(batch_size)]
        trees = [[] for _ in range(batch_size)]
        log_energies_for_idxing = log_energies.numpy()

        for _ in range(M):

            max_idx = tf.convert_to_tensor(np.argmax(np.reshape(log_energies_for_idxing, (batch_size, -1)), axis=-1), dtype=tf.int32)
            
            referred_by, referred_to, rel_type = self.flat_idx_to_tuple(max_idx, dimensions=[M, n_rel_types])
        
            for i in range(batch_size):
                link_from, link_to, rel_t = int(referred_by[i].numpy()), int(referred_to[i].numpy()), int(rel_type[i].numpy())
                
                if log_energies_for_idxing[i, link_from, link_to, rel_t]==-np.inf:
                    continue
                
                mst_energies[i] += log_energies[i, link_from, link_to, rel_t]
                
                for elem in partitions[i][link_from]:
                    log_energies_for_idxing[i, elem, partitions[i][link_to], :] = -np.inf
                
                for elem in partitions[i][link_to]:
                    log_energies_for_idxing[i, elem, partitions[i][link_from], :] = -np.inf
                
                #For ensuring every element can connect to only 1 other element
                log_energies_for_idxing[i, link_from, :, :] = -np.inf

                trees[i].append((link_from, link_to, rel_t))
                
                temp = partitions[i][link_from]+partitions[i][link_to]
                partitions[i][link_from] = partitions[i][link_to] = temp
            
            #log_energies[i, partitions[i][link_from], partitions[i][link_to], :] = tf.constant(-np.inf) for all i            
            #log_energies = tf.reshape(log_energies, (batch_size, -1))
            #log_energies = tf.where(tf.range(M*M*n_rel_types)==tf.reshape(max_idx, (-1, 1)), tf.constant(-np.inf), log_energies)
            #log_energies = tf.reshape(log_energies, (batch_size, M, M, n_rel_types))

        return tf.stack(mst_energies), trees
    
    @convert_tensor_to_lists(indices=(2,))
    def score_tree(self, log_energies: tf.Tensor, tree: List[List[Tuple[int,int,int]]]) -> tf.Tensor:
        """Calculates the log energies of a given batch of trees.
        Args:
            log_energies:   same, as in self.mst()
            tree:           A list of [list of M tuples of form (link_from, link_to, relation_type)] for each sample i in the batch (where M is the max. number of componenets in any sample of the batch)
        Returns:
            A tensor of size [batch_size] having the score of each tree corresponding to each sample(thread) of a batch.
        """
        scores = []
        
        for i in range(tf.shape(log_energies).numpy()[0]):
            tree_score = 0
            for (link_from, link_to, rel_type) in tree[i]:
                if not link_from==link_to==rel_type==config['pad_for']['refers_to_and_type']:
                    tree_score += log_energies[i, link_from, link_to, rel_type]
            scores.append(tree_score)
        
        return tf.stack(scores)
    
    @convert_tensor_to_lists(indices=(2,))
    def disc_loss(self, log_energies: tf.Tensor, label_tree: List[List[Tuple[int,int,int]]]) -> tf.Tensor:
        """Calculates average loss of a batch of samples.
        Args:
            log_energies:   same, as in self.mst()
            label_tree:     same, as in self.score_tree() [Labels for the actual thread of the tree]
        Returns:
            Average discrimnation loss(loss with partition function estimated with the tree with maximum energy).
        """
        mst_energies, _ = self.mst(log_energies)
        label_tree_scores = self.score_tree(log_energies, label_tree)
        return tf.reduce_mean(mst_energies-label_tree_scores)