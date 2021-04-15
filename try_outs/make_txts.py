import bs4
import os
import re
from bs4 import BeautifulSoup

def add_tags(post, user_dict):
    if post['author'] not in user_dict:
        user_dict[post['author']] = len(user_dict)
    text = str(post) 
    text = '[USER'+str(user_dict[post['author']])+'] '+text
    
    pattern0 = r'(\n\&gt; \*Hello[\S]*)'
    pattern1 = r"(https?://)(\s)*(www\.)?(\s)*((\w|\s)+\.)*([\w\-\s]+/)*([\w\-]+)((\?)?[\w\s]*=\s*[\w\%&]*)*(\.html|\.htm)*"
    pattern2 = r"\&gt;(.*)\n"

    text = text.replace('</claim>', '</claim> ')
    text = text.replace('<claim', ' <claim')
    text = text.replace('<premise', ' <premise')
    text = text.replace('</premise>', '</premise> ')
    text = re.sub(pattern0, '', text)                             #Replace Footnotes
    text = re.sub(pattern1, '[URL]', text)                        #Replace [URL] 
    text = re.sub(pattern2, '[STARTQ]'+r'\1'+' [ENDQ] ', text)    #Replace quoted text
    
    text = BeautifulSoup(str(text), "lxml").get_text()
    print(text)
    return text

if __name__=='__main__':
    for t in ['negative', 'positive']:        
        root = 'AmpersandData/change-my-view-modes/v2.0/'+t+'/'
        for f in os.listdir(root):
            
            filename = os.path.join(root, f)
            if not(os.path.isfile(filename) and f.endswith('.xml')):
                continue
            
            with open(filename, 'r') as g:
                xml_str = g.read()
            parsed_xml = BeautifulSoup(str(BeautifulSoup(xml_str, "lxml")), "xml")
            
            user_dict = dict()
            str_to_write = BeautifulSoup(str(parsed_xml.find('title').find('claim').contents[0]), 'lxml').get_text().strip()+'\n'
            for post in [parsed_xml.find('op')]+parsed_xml.find_all('reply'):
                str_to_write += add_tags(post, user_dict)+'\n'
            write_root = 'AmpersandData/ModifiedThreads'
            with open(os.path.join(write_root, f[:-4]+'_'+t+'.txt'), 'w') as g:
                g.write(str_to_write)