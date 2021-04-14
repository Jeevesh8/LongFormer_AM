import bs4
import os
from bs4 import BeautifulSoup
from typing import List
import glob

def get_claims_premise(parsed_xml: bs4.BeautifulSoup) -> (List[str], List[str]):
    """Returns the list of all claims, and another list of all premises in a post
    """
    claims_lis, premise_lis = [str(parsed_xml.find('title').find('claim').contents[0]).strip()], []

    for post in [parsed_xml.find('OP')]+parsed_xml.find_all('reply'):
        for elem in post.contents:
            elem = str(elem)
            
            if elem.startswith('<claim'):
                parsed_claim = BeautifulSoup(elem, "xml")
                claims_lis.append(str(parsed_claim.find('claim').contents[0]).strip())
            
            elif elem.startswith('<premise'):
                parsed_premise = BeautifulSoup(elem, "xml")
                premise_lis.append(str(parsed_premise.find('premise').contents[0]).strip())
    
    return claims_lis, premise_lis        

def get_elem_nos(parsed_xml: bs4.BeautifulSoup, claims: List[str], premises: List[str]):
    """
    Returns a dictionary with keys as ids of claim and premises and the values as the 
    idx of that claim/premise in entire post. The non-argumentative components are also counted
    to find the index of a claim/premise, but they are not added to the dictionary.
    """
    elem_nos = dict()
    cur_elem = 0
    elem_nos['title'] = cur_elem
    cur_elem += 1
    for post in [parsed_xml.find('OP')]+parsed_xml.find_all('reply'):
        for elem in post.contents:
            elem = str(elem)
            
            if elem.startswith('<claim'):
                parsed_claim = BeautifulSoup(elem, "xml")
                claim_id = str(parsed_claim.find('claim')['id'])
                elem_nos[claim_id] = cur_elem

            elif elem.startswith('<premise'):
                parsed_premise = BeautifulSoup(elem, "xml")
                premise_id = str(parsed_premise.find('premise')['id'])
                elem_nos[premise_id] = cur_elem
            
            cur_elem+=1
    return elem_nos
        
def add_distance_relation(parsed_xml: bs4.BeautifulSoup, claims: List[str], premises: List[str]):
    """Appends the id and relation:distance to each of claim and premise. All components are tab separated.
    Args:
        parsed_xml: the output of parsing the entire xml string of a .xml file in AMPERSAND
        claims: List of all the claims in a thread,
        premise: List of all the premises in a thread.
    """
    elem_nos = get_elem_nos(parsed_xml, claims, premises)
    title_text = str(parsed_xml.find('title').find('claim').contents[0]).strip()
    try:
        idx = claims.index(title_text)
    except:
        raise AssertionError("Title claim not found ! In thread with claims and premises: ", claims, premises)
    claims[idx] = 'title'+'\t'+'None:None'+'\t'+claims[idx]

    for post in [parsed_xml.find('OP')]+parsed_xml.find_all('reply'):
        
        for elem in post.contents:
        
            elem = str(elem)
            
            if elem.startswith('<claim'):
                parsed_claim = BeautifulSoup(elem, "xml")
                claim = str(parsed_claim.find('claim').contents[0]).strip()
                try:
                    idx = claims.index(claim)
                except:
                    raise AssertionError("Claim : ", claim, " not found in existing claims: ", claims)
                
                claim_id = str(parsed_claim.find('claim')['id'])
                try:
                    ref_id = str(parsed_claim.find('claim')['ref'])
                    difference = ref_id #elem_nos[ref_id]-elem_nos[claim_id]
                    claims[idx] = claim_id+'\t'+str(difference)+':'+str(parsed_claim.find('claim')['rel'])+'\t'+claims[idx]
                except:
                    claims[idx] = claim_id+'\t'+'None:None'+'\t'+claims[idx]
                
            elif elem.startswith('<premise'):
                parsed_premise = BeautifulSoup(elem, "xml")
                premise = str(parsed_premise.find('premise').contents[0]).strip()
                try:
                    idx = premises.index(premise)
                except:
                    raise AssertionError("Premise : ", premise, " not found in existing premises: ", premises)
                
                premise_id = str(parsed_premise.find('premise')['id'])
                try:
                    ref_id = str(parsed_premise.find('premise')['ref'])
                    difference = ref_id #elem_nos[ref_id]-elem_nos[premise_id]
                    premises[idx] = premise_id+'\t'+str(difference)+':'+str(parsed_premise.find('premise')['rel'])+'\t'+premises[idx]
                except:
                    premises[idx] = premise_id+'\t'+'None:None'+'\t'+premises[idx]

if __name__=='__main__':
    
    for t in ['negative', 'positive']:
        
        root = 'AmpersandData/change-my-view-modes/v2.0/'+t+'/'
        for f in os.listdir(root):
            filename = os.path.join(root, f)
            if os.path.isfile(filename) and f.endswith('.xml'):
                
                with open(filename, 'r') as g:
                    xml_str = g.read()
                
                parsed_xml = BeautifulSoup(xml_str, "xml")
                claims, premises = get_claims_premise(parsed_xml)
                
                with open(os.path.join('AmpersandData/Threads', f[:-4]+'_'+t+'.claim'), 'w') as g:
                    for claim in claims:
                        g.write(claim+'\n')    
                
                with open(os.path.join('AmpersandData/Threads', f[:-4]+'_'+t+'.premise'), 'w') as g:
                    for premise in premises:
                        g.write(premise+'\n')
                
                add_distance_relation(parsed_xml, claims, premises)
                
                with open(os.path.join('AmpersandData/ModifiedThreads', f[:-4]+'_'+t+'.claim'), 'w') as g:
                    for line in claims:
                        g.write(line+'\n')
                
                with open(os.path.join('AmpersandData/ModifiedThreads', f[:-4]+'_'+t+'.premise'), 'w') as g:
                    for line in premises:
                        g.write(line+'\n')
                
            
            