from configs import config

def labels_to_tags(labels):
    """Converts integer labels to component type labels.
    Input: array of integer labels 
    Output: arry of string tags
    """
    tag_dict = {v:k for k, v in config['arg_components'].items()}
    tag_dict[len(tag_dict)] = 'pad'
    return [tag_dict[i] for i in labels]

def get_rel_type_idx(relation: str) -> int:
    for i, v in enumerate(config['relations_map'].values()):
        if relation in v:
            return i
    return 0                                                                #Assuming None relation is 0-th position, always.