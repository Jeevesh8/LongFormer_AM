MAX_TOKEN_DIST = 
MIN_TOKEN_DIST = 
    
config = {
    'max_rel_length' : 2+MAX_TOKEN_DIST-MIN_TOKEN_DIST, # 0 for no token to refer. 1 for title. Rest for various distances.
    'dist_to_label'  : { i:j+2 for j, i in enumerate(range(MIN_TOKEN_DIST, MAX_TOKEN_DIST)) },
    'relations' : ['partial_attack', 'agreement', 
                   'attack', 'rebuttal_attack', 
                   'understand', 'undercutter', 
                   'undercutter_attack', 'disagreement', 
                   'rebuttal', 'support', 
                   'cont', 'None', 
                   'partial_agreement', 'partial_disagreement']
    'arg_components' : {'other': 0, 'B-C' : 1, 'I-C' : 2, 'B-P' : 3, 'I-P' : 4},
    'max_tokenizer_length': 8192+2,
    'max_rel_comps' : 5,                                                                        #The maximum number of components a component can be related to
}