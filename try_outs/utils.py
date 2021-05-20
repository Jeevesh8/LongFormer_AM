def find_sub_list(sl,l):
    '''
    Returns the start and end positions of sublist sl in l
    '''
    sll=len(sl)
    for ind in (i for i,e in enumerate(l) if e==sl[0]):
        if l[ind:ind+sll]==sl:
            return ind,ind+sll-1