from transformers import LongformerTokenizer
from typing import List, Dict

from component_generator import generate_components

def get_tokenizer():
    max_pos = 8192+2
    tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
    vocab_file, merges_file = tokenizer.save_vocabulary('.')
    tokenizer = LongformerTokenizer(vocab_file, merges_file, tokenizer.init_kwargs.update({'model_max_length' : max_pos}))
    with open('special_tokens.txt') as f:
        sp_tokens = list(filter(None, f.read().split('\n')))
        print(sp_tokens)
    tokenizer.add_tokens(sp_tokens)
    return tokenizer

def find_last_to_last(lis, elem_set)-> int:
    """Returns the index of last to last occurance of any element of elem_set in lis,
    if element is not found at least twice, returns -1. """
    count = 0
    for idx, elem in reversed(list(enumerate(lis))):
        if elem in elem_set:
            count+=1
        if count==2:
            return idx
    return 0

def get_tokenized_thread(filename)-> Tuple[List[str], Dict[str, int], Dict[str, int], Dict[str, Tuple[str, str]]]:
    """Returns the tokenized version of thread present in filename
    file must be an xml file of Ampersand data."""
    tokenizer = get_tokenizer()
    user_token_indices = [tokenizer.encode('[USER0]')[1], tokenizer.encode('[USER1]')[1]]

    begin_positions = dict()
    prev_comment_begin_position = dict()
    ref_n_rel_type = dict()

    tokenized_thread = [tokenizer._convert_token_to_id('<s>')]
    for component_tup in generate_components():
        component, comp_type, comp_id, refers, rel_type = component_tup
        encoding = tokenizer.encode(component)[1:-1]
        if comp_type=='claim' or comp_type=='premise':
            begin_positions[comp_id] = len(tokenized_thread)
            prev_comment_begin_position[comp_id] = find_last_to_last(tokenized_thread, user_token_indices)
            ref_n_rel_type[comp_id] = (refers, rel_type)
        tokenized_thread+=encoding
    tokenized_thread.append(tokenizer._convert_token_to_id('</s>'))
    
    return tokenized_thread, begin_positions, prev_comment_begin_position, ref_n_rel_type
