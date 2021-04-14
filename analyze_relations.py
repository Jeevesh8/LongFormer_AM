import bs4
import os
from bs4 import BeautifulSoup

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
        with open(os.path.join(root, f)) as g:
            lines = g.readlines()
            for line in lines:
                id, rel, _ = line.strip().split('\t')
                rel_types.append(rel.split(':')[1])
            
    print("All relation types : ", set(rel_types))
            
    two_post_farther, closer_than_two_post = 0, 0
    
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
