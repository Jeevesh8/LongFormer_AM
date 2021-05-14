from tokenize_components import get_tokenized_thread
from configs import config

def get_label_component_list(filename):
    """Returns a list of tuples of form
    (
    start position: int, 
    end position: int, 
    component type: int,                                 [config['arg_components']['B-C] or config['arg_components']['B-P']]
    refers to: List[Union[int, str]],                    [the beginning indices of the components that the current components refers to; "None" if the current component refers to no other component]
    relation type: str                                   [the type of relation. "None" if not related to anything.]
    )
    There is one tuple per argumentative component(claim/premise) in the xml file given by "filename".
    """
    tokenized_thread, begin_positions, prev_comment_begin_position, ref_n_rel_type, end_positions, comp_types = get_tokenized_thread(filename)
    real_comps = []
    
    for k in begin_positions.keys():
        begin = begin_positions[k]
        end = end_positions[k]
        component_type = config['arg_components']['B-C'] if comp_types[k] == 'claim' else config['arg_components']['B-P']
        refers_to = [elem if elem =='None' else begin_positions[elem] for elem in str(ref_n_rel_type[k][0]).split('_')]
        rel_type = str(ref_n_rel_type[k][1])
        real_comps.append((begin, end, component_type, refers_to, rel_type))
    
    return real_comps

def get_prev_comment_begin_positions(filename):
    """Returns a dictionary mapping the begin positions of various argumentative components
    to the begin positions of their previous comments, in the tokenized thread obtained from the 
    xml of "filename".
    """
    tokenized_thread, begin_positions, prev_comment_begin_position, ref_n_rel_type, end_positions, comp_types = get_tokenized_thread(filename)
    new_prev_comment_begin_positions = {}
    for k, v in begin_positions.items():
        new_prev_comment_begin_positions[v] = prev_comment_begin_position[k]
    return new_prev_comment_begin_positions, begin_positions

def get_prev_comment_begin_position(begin, prev_comment_begin_positions):
    """
    Args:
        begin: begin position of the argumentative component in the predictions, whose previous comment's begin position is to be found.
        prev_comment_begin_positions: A dicitionary mapping the various begin positions in the labels to the begin positions of their previous comments.
    Returns:
        The beginning position of the estimated previous comment of the current argumentative component.
        The estimated previous comment is obtained as the previous comment of the nearest argumentative component available in the label, 
        beginning before or at the same location as the argumentative component in the predictions.
    """
    begin_positions = list(prev_comment_begin_positions.keys())
    begin_positions.sort()
    prev_comment_begin_position = 0
    for elem in begin_positions:
        if elem<=begin:
            prev_comment_begin_position = prev_comment_begin_positions[elem]
        else:
            break
    return prev_comment_begin_position

def get_begin_from_refers(relative_dist_label, prev_comment_begin_position):    
    """Converts the prediction label for distance of the component, the current compoenent is related to relative to the previous comment's beginning,
    to the index in the tokenized thread where the the related component begins.
    Args:
        relative_dist_label: The relative distance label predicted by the model.
        prev_comment_begin_position: The index in the tokenized thread at which the previous comment begins.
    Returns:
        The begin index of the related component that is at distance given by "relative_dist_label" from the point specified in prev_comment_begin_position.
    """
    if relative_dist_label==0:
        return 'None'
    label_to_dist = {v:k for k,v in config['dist_to_label'].items()}
    relative_dist = label_to_dist[relative_dist_label]
    begin_idx_of_related_component =  prev_comment_begin_position+relative_dist
    if begin_idx_of_related_component<0:
        print("The begin index of related component is coming out to be negative in the predictions!! Previous comment begin position: ", prev_comment_begin_position, " & The relative distance predicted: ", relative_dist)
    return begin_idx_of_related_component

def get_pred_component_list(seq_length, filename, viterbi_seq, refers_preds, relation_type_preds):
    """Returns a list of tuples of the form specified in the get_component_list() function above, for the viterbi_seq predicted by the model.
    Args:
        seq_length: The length of the sequence predicted by the model.
        filename: The filename of the file containing the actual xml, used for getting previous comment's beginning positions and converting relative distances to absolute ones.
        viterbi_seq: List of ints corresponding to arg components labels of each token.
        refers_preds: List of ints corresponding to the token referred to by each token in the tokenized thread.
        relation_type_preds: List of ints corresponding to the index of relation type in config['relations']
    """
    predicted_components = []
    prev_comment_begin_positions, begin_positions = get_prev_comment_begin_positions(filename)
    j=0
    
    #Correct initial components
    if viterbi_seq[j]==config['arg_components']['I-C']: 
        viterbi_seq[j] = config['arg_components']['B-C'] if viterbi_seq[j+1]==config['arg_components']['I-C'] else config['arg_components']['other']
    if viterbi_seq[j]==config['arg_components']['I-P']:
        viterbi_seq[j] = config['arg_components']['B-P'] if viterbi_seq[j+1]==config['arg_components']['I-P'] else config['arg_components']['other']
    
    while j<seq_length:
        if viterbi_seq[j]==config['arg_components']['B-C']:
            begin = j
            j+=1
            while j<seq_length and viterbi_seq[j]==config['arg_components']['I-C']:
                j+=1
            end = j
            
            component_type = config['arg_components']['B-C']
            prev_comment_begin_position = get_prev_comment_begin_position(begin, prev_comment_begin_positions)
            refers_to = begin_positions['title'] if refers_preds[begin]==1 else get_begin_from_refers(refers_preds[begin], prev_comment_begin_position)
            rel_type = config['relations'][relation_type_preds[begin]]
            
            predicted_components.append((begin, end, component_type, refers_to, rel_type))
        
        elif viterbi_seq[j]==config['arg_components']['B-P']:
            begin = j
            j+=1
            while j<seq_length and viterbi_seq[j]==config['arg_components']['I-P']:
                j+=1
            end = j
            
            component_type = config['arg_components']['B-P']
            prev_comment_begin_position = get_prev_comment_begin_position(begin, prev_comment_begin_positions)
            refers_to = begin_positions['title'] if refers_preds[begin]==1 else get_begin_from_refers(refers_preds[begin], prev_comment_begin_position)
            rel_type = config['relations'][relation_type_preds[begin]]
            
            predicted_components.append((begin, end, component_type, refers_to, rel_type))
        
        else:
            while j<seq_length and viterbi_seq[j]==config['arg_components']['other']:
                j+=1
    
    return predicted_components

def change_input_dtypes(func):
    def new_func(filename, seq_length, viterbi_seq, refers_preds, relation_type_preds):
        for i in range(len(filename)):
            func(filename[i].numpy().decode('utf-8'), int(seq_length[i].numpy()), viterbi_seq[i].numpy().tolist(), refers_preds[i].numpy().tolist(), relation_type_preds[i].numpy().tolist())
    return new_func

@change_input_dtypes
def single_sample_eval(filename, seq_length, 
                       viterbi_seq, refers_preds, 
                       relation_type_preds,):
    """Prints out the evaluation results for a single sample thread.
    Args:
        filename: The xml file having the thread to be evaluated. (str)
        seq_length: The length of the sequence predicted with the crf. (int)
        viterbi_seq: The viterbi decoded sequence output by the crf. (tensor of ints with shape [None])
        refers_preds: The labels predicted for refering to other components by the model. (tensor of ints with shape [None])
        relation_type_preds: The labels predicted for the type of relations between components by the model. (tensor of ints with shape [None])
    Returns:
        None
    """
    print("Args: " , filename, seq_length, viterbi_seq, refers_preds, relation_type_preds)
    label_list = get_label_component_list(filename)
    preds_list = get_pred_component_list(seq_length, filename, viterbi_seq, refers_preds, relation_type_preds)
    print("Labels list: ", label_list)
    print("Predictions list: ", preds_list)
    
    component_label_list = [elem[:3] for elem in label_list]
    component_preds_list = [elem[:3] for elem in preds_list]
    
    #Component Types
    total_claims, correct_claims, total_premises, correct_premises = 0,0,0,0
    correct_pred_components, correct_label_components = [], []
    for j, elem in enumerate(component_preds_list):
        if elem[2]==config['arg_components']['B-C']:
            total_claims+=1
            try:
                idx = component_label_list.index(elem)
            except ValueError:
                continue
            correct_claims+=1
            correct_pred_components.append(preds_list[j])
            correct_label_components.append(label_list[idx])

        elif elem[2]==config['arg_components']['B-P']:
            total_premises+=1
            try:
                idx = component_label_list.index(elem)
            except ValueError:
                continue
            correct_premises+=1
            correct_pred_components.append(preds_list[j])
            correct_label_components.append(label_list[idx])
        
        else:
            raise ValueError("Can't find component type ", elem[2], " in argumentative component types:  [", config['arg_components']['B-C'], ", ", config['arg_components']['B-P'])

    #Referred components & Relation Types
    all_links, valid_links, valid_rel_types = 0, 0, 0
    for j, elem in enumerate(correct_pred_components):
        all_links+=1
        related_to_correct_component = False
        for comp in correct_pred_components:
            if elem[3]==comp[0] or elem[3]=='None' or elem[3]=='title':
                related_to_correct_component = True
                break
        
        if related_to_correct_component and elem[3] in correct_label_components[j][3]:
            valid_links+=1
            if elem[4]==correct_label_components[j][4]:
                valid_rel_types+=1
    
    print("Total Claims: ", total_claims)
    print("Total Premises: ", total_premises)
    print("Correct Claims: ", correct_claims)
    print("Correct Premises: ", correct_premises)
    print("Total links between components correct-ly predicted. Including None links: ", all_links)
    print("Correct links between componenets correctly predicted: ", valid_links)
    print("Correct Links between correct components with correct types: ", valid_rel_types)

