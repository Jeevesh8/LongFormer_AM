import bs4
import os
from bs4 import BeautifulSoup
from typing import List, Optional

def add_elem(elem: bs4.BeautifulSoup, claims_lis: List[str], premise_lis : List[str], to_claim: Optional[bool]=None):
    """Adds elem to claims_lis / premise_lis recursively. If the elem is non-argumentative, the function just returns
    Args:
        elem:   parsed xml of an outer most <claim>/<premise> component. This function will look through all the 
                <claim>/<premise> tags nested inside the outer-most one. The words are added to premise_lis/claims_lis
                based on which is the innermost tag that encloses the words.
        claims_lis: This is the list that is filled with all the claims in the current elem. 
        premise_lis: This is the list that is filled with all the premises in the current elem. 
        to_claim: Used internally. If True, then elem will be added to claims_lis. If False, elem will be added to premise_lis.
                  If None, elem is non-argumentative, and will be discarded.
    Returns:
        None
    """
    if str(elem).strip()=='':
        return
    
    if str(elem).strip().startswith('<claim'):
        elem = str(elem)
        parsed_claim = BeautifulSoup(elem, "xml")
        for part in parsed_claim.find('claim').contents:
            add_elem(part, claims_lis, premise_lis, True)
            
    elif str(elem).strip().startswith('<premise'):
        elem = str(elem)
        parsed_premise = BeautifulSoup(elem, "xml")
        for part in parsed_premise.find('premise').contents:
            add_elem(part, claims_lis, premise_lis, False)
    
    elif to_claim is None:
        return
    
    elif to_claim:
        elem = BeautifulSoup(str(elem), "lxml")
        claims_lis.append(elem.get_text().strip())
    
    else:
        elem = BeautifulSoup(str(elem), "lxml")
        premise_lis.append(elem.get_text().strip())
    
def get_claims_premise(parsed_xml: bs4.BeautifulSoup) -> (List[str], List[str]):
    """Returns the list of all claims, and another list of all premises in a post
    """
    claims_lis, premise_lis = [str(parsed_xml.find('title').find('claim').contents[0]).strip()], []

    for post in [parsed_xml.find('OP')]+parsed_xml.find_all('reply'):
        for elem in post.contents:
            add_elem(elem, claims_lis, premise_lis)        
    
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

def update_claims_premises(elem: bs4.BeautifulSoup, claims:List[str], premises:List[str], to_claim:Optional[bool]=None, parent_id:Optional[str]=None, parent_ref:Optional[str]=None, parent_rel:Optional[str]=None):
    """Adds ids, relationional info(ref_id:rel_type) extracted from elem into corresponding elements of claims and premises.
    Args:
        elem:   parsed xml of an outer most <claim>/<premise> component. This function will look through all the 
                <claim>/<premise> tags nested inside the outer-most one. The words are made to correspond to 
                the innermost tag that encloses the words. The parts thus obtained at a particular level of nesting, 
                sharing the same parent and without an id are chained together by using a new type of relation "cont"(for continue)
                and are assigned distinct ids.
        claims :  List of all claims without the relational information and ids.
        premises: List of all premises without the relational information and ids.
        to_claim: Used internally. If True, then elem will be added to claims. If False, elem will be added to premise. 
                  If None, elem is non-argumentative, and will be discarded.
        parent_id: Used internally. Id of the parent tag.
        parent_ref: Used internally. The element to which the parent tag is related.
        parent_rel: Used internally. The relation type between the parent and the element to which the parent tag is related.

        ----IDENTIFYING PARENTS & CHILDREN EXAMPLE----
        <claim id> I am sincerely <premise id1> your </premise> beloved. </claim>
        is transformed to -->
        <claim id> I am sincerely </claim> <premise id1> your </premise> <claim id+c, ref=id, rel=cont> beloved. </claim>
        Now, the parent of "I am sincerely" is the <claim id> tag, the parent of "your" is <premise id1> tag & the parent of 
        "beloved" <claim id+c...> tag.
        ----------------------------------------------
    
    Returns:
        None
    """
    if str(elem).strip()=='':
        #print("Returning Useless..")
        return 
    
    if str(elem).strip().startswith('<claim'):
        elem = str(elem).strip()
        parsed_claim = BeautifulSoup(elem, "xml")
        claim_id = str(parsed_claim.find('claim')['id'])
        try:
            ref_id = str(parsed_claim.find('claim')['ref'])
            difference = ref_id #elem_nos[ref_id]-elem_nos[claim_id]
            claim_ref, claim_rel = str(difference), str(parsed_claim.find('claim')['rel'])
        except:
            claim_ref, claim_rel = 'None', 'None'
               
        for part in parsed_claim.find('claim').contents:
            if not str(part).startswith('<claim') and not str(part).startswith('<premise') and not part==parsed_claim.find('claim').contents[0]:
                claim_ref = claim_id
                claim_id += 'c'                     # c for child/continue
                claim_rel = 'cont'
            update_claims_premises(part, claims, premises, True, claim_id, claim_ref, claim_rel)
    
    elif str(elem).strip().startswith('<premise'):
        elem = str(elem).strip()
        parsed_premise = BeautifulSoup(elem, "xml")
        premise_id = str(parsed_premise.find('premise')['id'])
        try:
            ref_id = str(parsed_premise.find('premise')['ref'])
            difference = ref_id #elem_nos[ref_id]-elem_nos[claim_id]
            premise_ref, premise_rel = str(difference), str(parsed_premise.find('premise')['rel'])
        except:
            premise_ref, premise_rel = 'None', 'None'
               
        for part in parsed_premise.find('premise').contents:
            if not str(part).startswith('<claim') and not str(part).startswith('<premise') and not part==parsed_premise.find('premise').contents[0]:
                premise_ref = premise_id
                premise_id += 'c'                     # c for child/continue
                premise_rel = 'cont'
            update_claims_premises(part, claims, premises, False, premise_id, premise_ref, premise_rel)
    
    elif to_claim is None:
        return
            
    elif to_claim:
        #print("Trying to add element: ", str(elem))
        elem = BeautifulSoup(str(elem), "lxml")
        claim = elem.get_text().strip()
        try:
            idx = claims.index(claim)
        except:
            raise AssertionError("Claim : ", claim, " not found in existing claims: ", claims)
        claims[idx] = parent_id+'\t'+parent_ref+':'+parent_rel+'\t'+claims[idx]
    
    else:
        elem = BeautifulSoup(str(elem).strip(), "lxml")
        premise = elem.get_text().strip()
        try:
            idx = premises.index(premise)
        except:
            raise AssertionError("Claim : ", claim, " not found in existing claims: ", claims)
        premises[idx] = parent_id+'\t'+parent_ref+':'+parent_rel+'\t'+premises[idx]

def add_distance_relation(parsed_xml: bs4.BeautifulSoup, claims: List[str], premises: List[str]):
    """Appends(to the beginning) the id and relation:distance to each of claim and premise. All components are tab separated.
    Args:
        parsed_xml: the output of parsing the entire xml string of a .xml file in AMPERSAND
        claims: List of all the claims in a thread,
        premise: List of all the premises in a thread.
    """
    elem_nos = get_elem_nos(parsed_xml, claims, premises)
    title_text = parsed_xml.find('title').find('claim').get_text().strip()
    try:
        idx = claims.index(title_text)
    except:
        raise AssertionError("Title claim not found ! In thread with claims and premises: ", claims, premises)
    claims[idx] = 'title'+'\t'+'None:None'+'\t'+claims[idx]

    for post in [parsed_xml.find('OP')]+parsed_xml.find_all('reply'):
        for elem in post.contents:
            update_claims_premises(elem, claims, premises)

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
                
            
            