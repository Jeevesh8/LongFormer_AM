import re

def clean_bot_response(text):
    '''
    Replace the bot response at the end
    of each submission with empty string.
    '''
    return re.sub(r'\n\&gt; \*Hello[\s\S]*', '', text)

def sub_url(text):
    '''
    Replace urls with special token [URL].
    '''
    regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
    return re.sub(regex, '[URL]', text)

def handle_quotes(text):
    '''
    Start and end each quote with special tokens
    [STARTQ] and [ENDQ]
    '''
    regex = r'\&gt;.*\n\n'
    for substr in re.findall(regex, text):
        _substr = " ".join(['[STARTQ]']+substr.split()[1:]+['[ENDQ]']) + " "
        text = text.replace(substr, _substr)
    return text

def clean_special_symbols(text):
    '''
    Replace \n, *, _, # with whitespace
    '''
    return re.sub(r'[\_|\*|\n|\#]+', " ", text)

def clean_pipeline(text, is_submission):
    if is_submission:
        return clean_special_symbols(handle_quotes(sub_url(clean_bot_response(text))))
    else:
        return clean_special_symbols(handle_quotes(sub_url(text)))
