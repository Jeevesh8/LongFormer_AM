import re
import glob
import json

def get_op(text):
    return re.findall(r'<OP[\s\S]+?</OP>', text)[0]
def get_replies(text):
    return re.findall(r'<reply[\s\S]+?</reply>', text)
def get_author(segment):
    return re.findall('<.*author.*?>', text)[0].split('author=')[-1][1:-2]
def get_claims(text):
    return re.findall(r'<claim[\s\S]+?</claim>', text)
def get_premises(text):
    return re.findall(r'<premise[\s\S]+?</premise>', text)
def strip_tags(text):
    return re.sub(r'<OP.*?>|</OP>|<claim.*?>|</claim>|<premise.*?>|</premise>|<reply.*?>|</reply>','',text)
def write_to_file(filename, text):
    with open(filename, 'w') as f:
        f.write(text)

for fname in ['./change-my-view-modes/v2.0/positive/19.xml', './change-my-view-modes/v2.0/negative/69.xml', './change-my-view-modes/v2.0/negative/70.xml', './change-my-view-modes/v2.0/negative/71.xml', './change-my-view-modes/v2.0/negative/72.xml']:
    write_fname = './Posts/'+fname.split('/')[-1][:-4]+'_'+fname.split('/')[-2]
    count = 0
    with open(fname, 'r') as f:
        text = f.read()
    op = get_op(text)
    op_author = get_author(op)
    op_text = strip_tags(op)
    op_claims = [strip_tags(claim) for claim in get_claims(op)]
    op_premises = [strip_tags(premise) for premise in get_premises(op)]

    write_to_file('_'.join([write_fname, str(count), '.txt']), op_text)
    write_to_file('_'.join([write_fname, str(count), '.claim']), '\n'.join(op_claims))
    write_to_file('_'.join([write_fname, str(count), '.premise']), '\n'.join(op_premises))

    for reply in get_replies(text):
        r_author = get_author(reply)
        r_text = strip_tags(reply)
        r_claims = [strip_tags(claim) for claim in get_claims(reply)]
        r_premises = [strip_tags(premise) for premise in get_premises(reply)]
        count += 1

        write_to_file('_'.join([write_fname, str(count), '.txt']), r_text)
        write_to_file('_'.join([write_fname, str(count), '.claim']), '\n'.join(r_claims))
        write_to_file('_'.join([write_fname, str(count), '.premise']), '\n'.join(r_premises))
