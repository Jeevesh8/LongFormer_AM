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

if __name__=='__main__':
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
                            parsed_claim = BeautifulSoup(elem, "xml")
                            try:
                                if len(parsed_claim.find('claim').contents)>1:
                                    nested_claim_premise+=1
                                    print("Claim : ", parsed_claim.find('claim').contents)
                            except:
                                if len(parsed_claim.find('premise').contents)>1:
                                    nested_claim_premise+=1
                                    print("Premise : ", parsed_claim.find('premise').contents)
                            try :
                                ref_id = str(parsed_claim.find('claim')['ref'])
                                if ref_id not in cur_claim_premise and ref_id not in prev_claim_premise:
                                    two_post_farther+=1
                                else:
                                    closer_than_two_post+=1
                            except:
                                continue
                    prev_claim_premise = cur_claim_premise
                
    print("Relations inside current reply or in previous post/title : ", closer_than_two_post)
    print("Relations not found inside current reply and in previous post/title : ", two_post_farther)
    print("Nested claims / premises: ", nested_claim_premise)
    
    replaces =  [('&', 'and'), ('’',"\'"), ('“', '\"'), ('”', '\"')]
    max_pos = 8192+2
    tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
    vocab_file, merges_file = tokenizer.save_vocabulary('.')
    tokenizer = LongformerTokenizer(vocab_file, merges_file, tokenizer.init_kwargs.update({'model_max_length' : max_pos}))
    
    root = 'AmpersandData/ModifiedThreads/'
    cnp_root = 'AmpersandData/ModifiedThreads/'
    
    distances = []
    
    for f in os.listdir(root):
        filename = os.path.join(root, f)
        if os.path.isfile(filename) and filename.endswith('.txt') and filename.find('._')==-1:
            with open(filename) as g:
                content = g.read()
            
            begin_positions = dict()
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
                        claim_id, dist_n_rel, claim = elem.split('\t')
                        claim = ' '.join(claim.split())
                        
                        for elem in replaces:
                            claim = claim.replace(*elem)
                        
                        i = 0
                        tokenized_claim = tokenizer.encode(claim)[1:-1]
                        #print("Sample claim encoding: ", len(tokenized_claim), tokenized_claim)
                        indices = find_sub_list(tokenized_claim, input_ids)
                        
                        if indices is None:
                            tokenized_claim = tokenizer.encode(' '+claim)[1:-1]
                            #print("Sample claim encoding: ", len(tokenized_claim), tokenized_claim)
                            indices = find_sub_list(tokenized_claim, input_ids)
                            
                        assert indices is not None, str(claim) + " in "+ filename #"Unable to find claim : "+ str(tokenizer.convert_ids_to_tokens(tokenized_claim)) + " in " + str(tokenizer.convert_ids_to_tokens(input_ids)) #str(claim) + " in "+ content
                        
                        begin_positions[claim_id] = indices[0]
                        
            for component_type in ['.claim', '.premise']:
                with open(os.path.join(cnp_root, f[:-4]+component_type)) as g:
                    claims = []
                    for elem in g.read().split('\n'):
                        if elem.strip()=='':
                            continue
                        claim_id, dist_n_rel, claim = elem.split('\t')
                        ref, rel_type = dist_n_rel.split(':')
                        if ref!='None':
                            for ref in ref.split('_'):
                                distances.append(begin_positions[ref]-begin_positions[claim_id])
                                if distances[-1]>0:
                                    #print(os.path.join(cnp_root, f[:-4]+component_type), ref, claim_id)
            
    print(max(distances), min(distances), distances)    