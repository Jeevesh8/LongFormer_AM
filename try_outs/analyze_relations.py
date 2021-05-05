import bs4
import os
from bs4 import BeautifulSoup
from transformers import LongformerTokenizer
from utils import find_sub_list

def get_cur_ids(post: str):
    ids = ['title']
    for elem in post.contents:
        elem = str(elem)
        if elem.startswith('<claim') or elem.startswith('<premise'):
            parsed_component = BeautifulSoup(elem, "xml")
            try:
                ids.append(str(parsed_component.find('claim')['id']))
            except:
                ids.append(str(parsed_component.find('premise')['id']))
    return ids

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

if __name__=='__main__':
    #Listing all different relation types
    root = 'AmpersandData/ModifiedThreads/' 
    rel_types = []
    for f in os.listdir(root):
        if f.endswith('claim') or f.endswith('.premise'):
            with open(os.path.join(root, f)) as g:
                lines = g.readlines()
                for line in lines:
                    id, rel, _ = line.strip().split('\t')
                    rel_types.append(rel.split(':')[1])
                
    print("All relation types : ", set(rel_types))

    far_away_ids = []
    #Calculating number of relations to posts beyond one previous post and number of nested claim/premise
    two_post_farther, closer_than_two_post = 0, 0
    nested_claim_premise = 0
    for t in ['negative', 'positive']:
        root = 'AmpersandData/change-my-view-modes/v2.0/'+t+'/'
        for f in os.listdir(root):
            filename = os.path.join(root, f)
            if os.path.isfile(filename) and f.endswith('.xml'):
                with open(filename, 'r') as g:
                    xml_str = g.read()
                parsed_xml = BeautifulSoup(xml_str, "xml")
                
                prev_claim_premise = ['title']
                
                for post in [parsed_xml.find('OP')]+parsed_xml.find_all('reply'):
                    cur_claim_premise = get_cur_ids(post)
                    for elem in post.contents:
                        elem = str(elem)
                        if elem.startswith('<claim') or elem.startswith('<premise'):
                            parsed_component = BeautifulSoup(elem, "xml")
                            try:
                                if len(parsed_component.find('claim').contents)>1:
                                    nested_claim_premise+=1
                                    print("Content Nested inside Claim : ", parsed_component.find('claim').contents)
                            except:
                                if len(parsed_component.find('premise').contents)>1:
                                    nested_claim_premise+=1
                                    print("Content Nested inside Premise : ", parsed_component.find('premise').contents)
                            try:
                                ref_ids = str(parsed_component.find('claim')['ref'])
                                component_id = str(parsed_component.find('claim')['id'])
                            except:
                                try:
                                    ref_ids = str(parsed_component.find('premise')['ref'])
                                    component_id = str(parsed_component.find('premise')['id'])
                                except:
                                    continue
                                
                            for ref_id in ref_ids.split('_'):
                                if ref_id not in cur_claim_premise and ref_id not in prev_claim_premise:
                                    two_post_farther+=1
                                    far_away_ids.append((f[:-4]+'_'+t, ref_id, component_id))
                                    #print("Two posts farther :", ref_id, " in ", str(parsed_component))
                                else:
                                    closer_than_two_post+=1
                            
                    prev_claim_premise = cur_claim_premise
                
    print("Relations inside current reply or in previous post/title[Excluding nested components] : ", closer_than_two_post)
    print("Relations not found inside current reply and in previous post/title[Excluding nested components] : ", two_post_farther)
    print("Nested claims / premises: ", nested_claim_premise)
    
    replaces =  [('&', 'and'), ('’',"\'"), ('“', '\"'), ('”', '\"')]
    max_pos = 8192+2
    tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
    vocab_file, merges_file = tokenizer.save_vocabulary('.')
    tokenizer = LongformerTokenizer(vocab_file, merges_file, tokenizer.init_kwargs.update({'model_max_length' : max_pos}))
    
    with open('special_tokens.txt') as f:
        sp_tokens = list(filter(None, f.read().split('\n')))
        print(sp_tokens)
    tokenizer.add_tokens(sp_tokens)
    user_token_indices = [tokenizer.encode('[USER0]')[1:-1][0], tokenizer.encode('[USER1]')[1:-1][0]]
    print("User tokens :", user_token_indices)
    
    root = 'AmpersandData/ModifiedThreads/'
    cnp_root = 'AmpersandData/ModifiedThreads/'
    
    distances = []
    dist_relative_to_prev_comment_beginning = []
    title_relations = 0

    for f in os.listdir(root):
        filename = os.path.join(root, f)
        if os.path.isfile(filename) and filename.endswith('.txt') and filename.find('._')==-1:
            with open(filename) as g:
                content = g.read()
            
            begin_positions = dict()
            prev_comment_begin_position = dict()
            prev_content = content
            
            content = '\n'.join([' '.join(elem.split()) for elem in content.split('\n')])
            for elem in replaces:
                content = content.replace(*elem)
            
            input_ids = tokenizer(content)['input_ids']
            #print("Input ids: ", len(input_ids), input_ids)
            
            for component_type in ['.claim', '.premise']:
                with open(os.path.join(cnp_root, f[:-4]+component_type)) as g:
                    for elem in g.read().split('\n'):
                        if elem.strip()=='':
                            continue
                        
                        component_id, dist_n_rel, component = elem.split('\t')
                        component = ' '.join(component.split())
                        
                        for elem in replaces:
                            component = component.replace(*elem)
                        
                        i = 0
                        tokenized_component = tokenizer.encode(component)[1:-1]
                        #print("Sample claim encoding: ", len(tokenized_component), tokenized_component)
                        indices = find_sub_list(tokenized_component, input_ids)
                        tokenized_component = tokenizer.encode(' '+component)[1:-1]
                        #print("Sample claim encoding: ", len(tokenized_component), tokenized_component)
                        alt_indices = find_sub_list(tokenized_component, input_ids)
                        
                        if indices is None or (alt_indices is not None and alt_indices[0]<indices[0]):
                            indices = alt_indices
                            
                        assert indices is not None, str(component) + " in "+ filename #"Unable to find claim : "+ str(tokenizer.convert_ids_to_tokens(tokenized_component)) + " in " + str(tokenizer.convert_ids_to_tokens(input_ids)) #str(component) + " in "+ content
                        
                        if f[:-4]=='302_negative' and component_id=='2':
                            print("302_negative: " , indices, component, len(input_ids))
                        
                        begin_positions[component_id] = indices[0]
                        prev_comment_begin_position[component_id] = find_last_to_last(input_ids[:indices[0]], user_token_indices)
                        
            for component_type in ['.claim', '.premise']:
                with open(os.path.join(cnp_root, f[:-4]+component_type)) as g:
                    components = []
                    for elem in g.read().split('\n'):
                        if elem.strip()=='':
                            continue
                        
                        component_id, dist_n_rel, component = elem.split('\t')
                        ref, rel_type = dist_n_rel.split(':')
                        for ref in ref.split('_'):
                            if ref not in ['None', 'title']:
                                distances.append(begin_positions[ref]-begin_positions[component_id])
                                dist_relative_to_prev_comment_beginning.append(begin_positions[ref]-prev_comment_begin_position[component_id])
                                if dist_relative_to_prev_comment_beginning[-1]<0:
                                    print("Very far away referenced: ", ref," from ", component_id)
                                    if (f[:-4], ref, component_id) not in far_away_ids:
                                        print(far_away_ids)
                                        print(begin_positions[ref], prev_comment_begin_position[component_id], begin_positions[component_id])
                                        if not(ref=='20' and component_id=='22' and f[:-4]=='11_positive'):
                                            raise AssertionError("Expected "+component_id+" to reference "+ref+" in "+f[:-4])
                            if ref=='title':
                                title_relations+=1
                        
    print(max(distances), min(distances), distances)
    print("Distances relative to beginning of previous comment(excluding relations to title) : ")
    print(max(dist_relative_to_prev_comment_beginning), min(dist_relative_to_prev_comment_beginning), dist_relative_to_prev_comment_beginning)
    print("Relations to title: ", title_relations)