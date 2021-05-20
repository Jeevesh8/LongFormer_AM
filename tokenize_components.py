import os
import tensorflow as tf
from typing import List, Dict, Tuple

from configs import config, tokenizer, user_token_indices
from component_generator import generate_components
from my_utils import convert_outputs_to_tensors
from simp_utils import get_rel_type_idx

def get_arg_comp_lis(comp_type, length):
    """Returns a list of labels for a component of comp_type of specified length."""
    assert comp_type in ["claim", "premise"], (
        "Un-supported component type: "
        + comp_type
        + " Try changing 'arg_components' in config.py"
    )
    comp_type = "C" if comp_type == "claim" else "P"
    begin = config["arg_components"]["B-" + comp_type]
    intermediate = config["arg_components"]["I-" + comp_type]
    return [begin] + [intermediate] * (length - 1)


def get_ref_link_lis(related_to, first_idx, last_idx) -> List[int]:
    """To be used to getting token ids to link the component between [first_index, last_index) to the component at distance related_to.
    Every token except the first, refers to its previous token.
    Args:
        related_to: The distance of the related component from(beginning of previous comment or any other place).
        first_idx:  The first index of the component whose tokens need to be linked.
    Returns: a List of the token positions that each token of the component beginning at first_idx refers to.
    """
    try:
        refs = [config["dist_to_label"][related_to]]
    except:
        refs = [0]
    return refs + [config["dist_to_label"][i] for i in range(first_idx, last_idx - 1)]


def get_global_attention(tokenized_thread, user_token_indices):
    global_attention = [0] * len(tokenized_thread)
    for i, elem in enumerate(tokenized_thread):
        if elem in user_token_indices:
            global_attention[i] = 1
    return global_attention


def find_last_to_last(lis, elem_set) -> int:
    """Returns the index of last to last occurance of any element of elem_set in lis,
    if element is not found at least twice, returns -1."""
    count = 0
    for idx, elem in reversed(list(enumerate(lis))):
        if elem in elem_set:
            count += 1
        if count == 2:
            return idx
    return 0


def get_tokenized_thread(
    filename,
) -> Tuple[
    List[str],
    Dict[str, int],
    Dict[str, int],
    Dict[str, Tuple[str, str]],
    Dict[str, int],
    Dict[str, str],
]:
    """Returns the tokenized version of thread present in filename. File must be an xml file of Ampersand data.
    Returns a tuple having:
        tokenized_thread: A 1-D integer List containing the input_ids output by tokenizer for the text in filename.
        begin_positions:  A dictionary mapping the ids of various argumentative components to their beginning index in tokenized_thread.
        ref_n_rel_type:  A dictionary mapping the ids of various argumentative components to a tuple of the form
                        ('_' separated string of all components the component relates to, relation type)

        end_positions: A dictionary mapping the ids of various argumentative components to their ending index+1 in tokenized_thread.
        comp_types: A dictionary mapping the ids of various argumentative components to their types ('claim'/'premise')
    """
    begin_positions = dict()
    end_positions = dict()
    ref_n_rel_type = dict()
    comp_types = dict()

    tokenized_thread = [tokenizer._convert_token_to_id("<s>")]
    for component_tup in generate_components(filename):
        component, comp_type, comp_id, refers, rel_type = component_tup
        encoding = tokenizer.encode(component)[1:-1]
        if comp_type in ["claim", "premise"]:
            begin_positions[comp_id] = len(tokenized_thread)
            end_positions[comp_id] = len(tokenized_thread) + len(encoding)
            ref_n_rel_type[comp_id] = (refers, rel_type)
            comp_types[comp_id] = comp_type
        tokenized_thread += encoding
    tokenized_thread.append(tokenizer._convert_token_to_id("</s>"))

    return (
        tokenized_thread,
        begin_positions,
        ref_n_rel_type,
        end_positions,
        comp_types,
    )


@convert_outputs_to_tensors(dtype=tf.int32)
def get_thread_with_labels(filename):
    """Returns the tokenized threads along with all the proper labels.
    Args:
        filename: The xml whose data is to be tokenized.

    Returns a tuple having:
        tokenized_thread: A 1-D integer List containing the input_ids output by tokenizer for the text in filename.
        comp_type_labels: A 1-D integer List containing labels for the type of component(other, begin-claim, inter-claim..) [size = len(tokenized_thread)]
        refers_labels:    A 2-D integer List containing labels for all the components the i-th token is linked to.       [size = [len(tokenized_thread), max_possible_links]]
        relation_type_labels: A 1-D integer List having the type of link that the i-th token has with the token it is related to. [size = [len(tokenized_thread)]]
        attention_mask:       A 1-D List of 1's.  [size = [len(tokenized_thread)]]
        global_attention:     A 1-D integer List having 1 where there is user tokens, 0 elsewhere. [size = [len(tokenized_thread),]]
    """
    (
        tokenized_thread,
        begin_positions,
        ref_n_rel_type,
        end_positions,
        comp_types,
    ) = get_tokenized_thread(filename)

    comp_type_labels = [config["arg_components"]["other"]] * len(tokenized_thread)
    attention_mask = [1] * len(tokenized_thread)
    global_attention = get_global_attention(tokenized_thread, user_token_indices)

    prev_end = 0
    refer_to_and_type = []
    
    begin_pos_lis = [v for _,v in begin_positions.items()]
    begin_pos_lis.sort()

    for comp_id in begin_positions:
        ref, rel = ref_n_rel_type[comp_id]
        begin, end = begin_positions[comp_id], end_positions[comp_id]

        assert prev_end <= begin, (
            "Overlapping components! End of previous component: "
            + str(prev_end)
            + " .Beginning of next component: "
            + str(begin)
        )
        assert 0 <= begin <= end < len(
            tokenized_thread
        ) or 0 <= prev_end + 1 <= begin <= end < len(
            tokenized_thread
        ), (
            "Begin, and end are not correct."
            + str(begin)
            + ", "
            + str(end)
        )

        comp_type_labels[begin:end] = get_arg_comp_lis(comp_types[comp_id], end - begin)
        
        rel_type = get_rel_type_idx(str(rel))    
        
        for j, ref_id in enumerate(str(ref).split("_")):
            if j==0:
                
                # skip self-referential components
                if ref_id==comp_id:
                    break
                
                elif ref_id == "None":
                    refer_to_and_type.append((begin_pos_lis.index(begin)+1, 0, rel_type))
                
                else:
                    refer_to_and_type.append((begin_pos_lis.index(begin)+1, begin_pos_lis.index(begin_positions[ref_id])+1, rel_type))
            else:
                print("Skipping the extra link for component: ", comp_id, " to ", ref_id, " for file: ", filename)
        
        prev_end = end

    assert (
        len(tokenized_thread)
        == len(comp_type_labels)
        == len(attention_mask)
        == len(global_attention)
    ), "Incorrect Dataset Loading !!"

    return (
        tokenized_thread,
        comp_type_labels,
        refer_to_and_type,
        attention_mask,
        global_attention,
    )


def get_model_inputs(file_lis):
    if type(file_lis) is str:
        assert os.path.isdir(
            file_lis
        ), "get_model_inputs() take either a directory name or file list as input! The provided argument is incorrect."
        file_lis = [os.path.join(file_lis, f) for f in os.listdir(file_lis)]
    for filename in file_lis:
        if not (os.path.isfile(filename) and filename.endswith(".xml")):
            continue
        yield (tf.constant(filename), *get_thread_with_labels(filename))
