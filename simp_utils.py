from cofigs import config

def labels_to_tags(labels):
    """Converts integer labels to component type labels.
    Input: array of integer labels 
    Output: arry of string tags
    """
    tag_dict = {v:k for k, v in config['arg_components']}
    return [tag_dict[i] for i in labels]
