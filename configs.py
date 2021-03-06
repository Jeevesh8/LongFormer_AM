from my_utils import get_tokenizer

MAX_TOKEN_DIST = 4000
MIN_TOKEN_DIST = 0

config = {
    'max_rel_length' : 2+MAX_TOKEN_DIST-MIN_TOKEN_DIST, # 0 for no token to refer. 1 for title. Rest for various distances.
    'dist_to_label'  : { i:j+2 for j, i in enumerate(range(MIN_TOKEN_DIST, MAX_TOKEN_DIST)) },
    'relations' : ['partial_attack', 'agreement', 
                   'attack', 'rebuttal_attack', 
                   'understand', 'undercutter', 
                   'undercutter_attack', 'disagreement', 
                   'rebuttal', 'support', 
                   'cont', 'None', 
                   'partial_agreement', 'partial_disagreement'],
    'arg_components' : {'other': 0, 'B-C' : 1, 'I-C' : 2, 'B-P' : 3, 'I-P' : 4},
    'max_tokenizer_length': 8192+2,
    'attention_window' : 192,
    'max_rel_comps' : 5,                                                                        #The maximum number of components a component can be related to
    'batch_size' : 5,
    'n_iters': 800,
}

tokenizer, user_token_indices = get_tokenizer(config['max_tokenizer_length'])

config['pad_for'] = {
    'tokenized_thread' : tokenizer.pad_token_id,
    'comp_type_labels' : config['arg_components']['other'], #len(config['arg_components']),
    'refers_labels' : 0, #len(config['dist_to_label'])+2,
    'relation_type_labels' : config['relations'].index('None'),
    'attention_mask' : 0,
    'global_attention_mask' : 0,
}