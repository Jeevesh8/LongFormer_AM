import tensorflow as tf
from typing import List

class tree_crf(tf.keras.layers.Layer):
    def __init__(self, prior):
        self.prior = prior
    
    def ss_sum(self, n_nodes: int, energies: tf.Tensor):
        """
        Args:
            n_nodes:    The number of nodes in the tree(i.e., the number of argumentative components+1(for None))
            energies:   Tensor of shape [N+1,N+1] padded to shape [M+1, M+1] {M = max. no. of argumentative components for any element in the batch}
        Returns:
            The sum of energies of all possible directed acyclic graphs(trees) over the N+1 nodes
        """
        memo = {}
        
        def recursive_ss_sum(assignments: tf.Tensor, to_assign: int, assignables: List[List[int]]):
            """
            Args:
                assignments:    A tensor of shape [M+1] denoting, the i-th entry denoting the commponent number referred to by the i-th component.
                                If the i-th entry is -1, it means that the i-th component's head hasn't been decided, yet.
                to_assign:      The index of the component to which a head is to be assigned in the current step of the recursion
            """
            if to_assign==n_nodes:
                return 1
            
            nonlocal energies, memo
            assignments_key = tf.io.serialize_tensor(assignments).numpy()
            
            try:
                return memo[assignments_key]
            except KeyError:
                energies_sum = 0
                for elem in assignables[to_assign]:
                    assignments[to_assign] = elem
                    if elem>to_assign:
                        assignables[elem].remove(to_assign)
                    energies_sum += energies[to_assign][elem]*recursive_ss_sum(assignments, to_assign+1, assignables)
                
                assignments[to_assign] = -1
                memo[assignments_key] = energies_sum
                return energies_sum

        return recursive_ss_sum(tf.constant([0]+[-1]*(n_nodes-1)), 1)