import json
import re
import torch
from transformers import (
    BertForMaskedLM,
    TFBertForMaskedLM,
    LongformerForMaskedLM,
    TFLongformerForMaskedLM,
    LongformerConfig,
    LongformerTokenizer,
)

# from cleaner import clean_pipeline
import glob
import os


def clear_directory(dir_name):
    """Delete all files from a given directory"""
    for filename in glob.glob(dir_name + "/*"):
        os.remove(filename)


def get_content_files(dir_name):
    """returns the list of text files within directory"""
    return glob.glob(dir_name + "/*.txt")


def create_mask_map(markers, tokenizer, special_token):
    """
    Given list of markers to mask, prepares a list of tuples
    (marker, <space separated mask tokens equal to the number of  subwords from marker>)
    """
    marker_map = []
    for marker in markers:
        masks = " ".join(
            [special_token for _ in range(len(tokenizer.encode(marker)) - 2)]
        )
        marker_map.append((marker, masks))
    return sorted(marker_map, key=lambda x: len(x[1].split()), reverse=True)


def mask_token_in_text(text, markers, tokenizer):
    """
    Mask tokens from given list of markers
    """
    marker_map = create_mask_map(markers, tokenizer, tokenizer.mask_token)
    for marker, mask in marker_map:
        regex = r"(?<=\W)%s(?=(\s|\W|$))" % marker
        text = re.sub(regex, mask, text, flags=re.IGNORECASE)
    return text


def get_marker_spans(text, markers):
    """
    Return character spans and marker indices in text
    """
    sorted_markers = sorted(markers, key=lambda x: len(x.split()), reverse=True)
    index = 1
    marked_spans = [0 for _ in text]
    for marker in sorted_markers:
        regex = r"(?:^|(?<=\W))%s(?=(\s|\W|$))" % marker
        match_iter = re.finditer(regex, text, flags=re.IGNORECASE)
        for m in match_iter:
            start, end = m.span()
            marked_spans[start:end] = [index for _ in range(start, end)]
        index += 1
    return marked_spans


def get_marker_features(text, markers, offset):
    """
    Given a text, return marker indices for each token position
    """
    marked_spans = get_marker_spans(text, markers)
    features = []
    for s, e in offset[1:-1]:
        if s < len(marked_spans):
            features.append(marked_spans[s])
        else:
            features.append(0)
    return [0] + features + [0]


def create_marker_token_list(markers, tokenizer):
    marker_ids = [tokenizer.encode(marker)[1:-1] for marker in markers]
    map_list = []
    for i in range(max([len(ids) for ids in marker_ids])):
        map_list.append(set([ids[i] for ids in marker_ids if len(ids) > i]))
    return map_list


def mask_tokens(encoded_text, markers, tokenizer):
    map_list = create_marker_token_list(markers, tokenizer)
    masked_text = []
    in_marker = False
    ix = 0
    for token in encoded_text:
        if not in_marker:
            if token in map_list[ix]:
                in_marker = True
                ix += 1
                masked_text.append(tokenizer.mask_token_id)
            else:
                masked_text.append(token)
        else:
            if ix < len(map_list):
                if token in map_list[ix]:
                    ix += 1
                    masked_text.append(tokenizer.mask_token_id)
                else:
                    in_marker = False
                    ix = 0
                    masked_text.append(token)
            else:
                in_marker = False
                ix = 0
                masked_text.append(token)
    return masked_text


def get_custom_model(max_pos, attention_window):
    """
    Loads pretrained model, resizes embedding (with PyTorch), save, load weights to Tensorflow version and return
    This is due to bug in Huggingface Transformers which causes shape mismatch between output and input embeddings
    of Tensorflow models when resizing
    """
    model = LongformerForMaskedLM.from_pretrained(
        "allenai/longformer-base-4096", attention_window=attention_window
    )
    config = model.config

    # extend position embeddings
    (
        current_max_pos,
        embed_size,
    ) = model.longformer.embeddings.position_embeddings.weight.shape
    max_pos += 2  # NOTE: RoBERTa has positions 0,1 reserved, so embedding size is max position + 2
    config.max_position_embeddings = max_pos
    assert max_pos > current_max_pos
    # allocate a larger position embedding matrix
    new_pos_embed = model.longformer.embeddings.position_embeddings.weight.new_empty(
        max_pos, embed_size
    )
    # copy position embeddings over and over to initialize the new position embeddings
    k = 2
    step = current_max_pos - 2
    while k < max_pos - 1:
        new_pos_embed[
            k : (k + step)
        ] = model.longformer.embeddings.position_embeddings.weight[2:]
        k += step
    model.longformer.embeddings.position_embeddings.weight.data = new_pos_embed
    model.longformer.embeddings.position_ids.data = torch.tensor(
        [i for i in range(max_pos)]
    ).reshape(1, max_pos)
    model.save_pretrained("tmp/LF/")

    tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
    vocab_file, merges_file = tokenizer.save_vocabulary(".")
    tokenizer = LongformerTokenizer(
        vocab_file,
        merges_file,
        tokenizer.init_kwargs.update({"model_max_length": max_pos}),
    )

    return (
        TFLongformerForMaskedLM.from_pretrained(
            "tmp/LF", from_pt=True, attention_window=attention_window
        ),
        tokenizer,
    )


def find_sub_list(sl, l):
    """
    Returns the start and end positions of sublist sl in l
    """
    sll = len(sl)
    for ind in (i for i, e in enumerate(l) if e == sl[0]):
        if l[ind : ind + sll] == sl:
            return ind, ind + sll - 1


def tokenize_and_label(annot_text, tokenizer):
    """
    Input: a dictionary of annotated segments and main text
    Output: a dictionary of encoded text, labels
    Labels: start_claim=1, cont_claim=2, start_premise=3, cont_premise=4, none=0
    """
    cleaned_text = clean_pipeline(annot_text["text"], False)
    encoded_text = tokenizer(cleaned_text, truncation=True, return_tensors="tf")
    enc_tokens = list(encoded_text["input_ids"].numpy()[0])
    labels = [0 for _ in enc_tokens]
    for claim in annot_text["claim"]:
        claim = clean_pipeline(claim, False)
        indices = find_sub_list(tokenizer.encode(claim)[1:-1], enc_tokens)
        if indices is not None:
            labels[indices[0]] = 1
            for i in range(indices[0] + 1, indices[1] + 1):
                labels[i] = 2
    for premise in annot_text["premise"]:
        premise = clean_pipeline(premise, False)
        indices = find_sub_list(tokenizer.encode(premise)[1:-1], enc_tokens)
        if indices is not None:
            labels[indices[0]] = 3
            for i in range(indices[0] + 1, indices[1] + 1):
                labels[i] = 4
    return encoded_text, labels


def labels_to_tags(labels):
    """
    Input: array of integer labels
    Output: arry of string tags
    """
    tag_dict = {
        0: "O",
        1: "B-claim",
        2: "I-claim",
        3: "B-premise",
        4: "I-premise",
        5: "PAD",
    }
    return [tag_dict[i] for i in labels]


def train_test_split():
    with open("../Data/annotated_threads.json", "r") as f:
        data = json.load(f)
    pos_keys = [key for key in data.keys() if "positive" in key]
    neg_keys = [key for key in data.keys() if "negative" in key]
    return pos_keys[:10] + neg_keys[:10], pos_keys[10:] + neg_keys[10:]


def labeled_thread_to_post(
    thread_dir, annotation_dir, post_dir, thread_to_post_map, seen_idfile
):
    """
    thread_dir is the directory containing unlabeled threads (to be annotated)
    annotation_dir is the directory containing .claim and .premise files generated by model for each thread in thread_dir
    post_dir is the target directory to save post text and annotations
    thread_to_post_map is the file containing thread_id to post_id mappings
    seen_ids contain post ids present in gold data
    """
    with open(seen_idfile) as f:
        seen_ids = list(filter(None, f.read().split("\n")))
    """
    thread_to_post maps thread file name (key) to the post ids (list)
    """
    thread_to_post = {}
    with open(thread_to_post_map) as f:
        maps = list(filter(None, f.read().split("\n")))
    for map in maps:
        thread_to_post[map.split("\t")[0]] = list(filter(None, map.split("\t")))[1:]

    def is_component_in_post(text, _claim, _premise):
        for c in filter(None, _claim.split("\n")):
            if c in text:
                return True
        for p in filter(None, _premise.split("\n")):
            if p in text:
                return True

    post_annotated = 0
    for threadfile in glob.glob(thread_dir + "/*.txt"):
        threadid = threadfile.split("/")[-1][:-4]
        with open(threadfile) as tfile:
            thread = tfile.read()
        with open(annotation_dir + "/" + threadid + ".claim") as claim_file:
            claims = claim_file.read()
        with open(annotation_dir + "/" + threadid + ".premise") as premise_file:
            premises = premise_file.read()
        with open(annotation_dir + "/" + threadid + ".score") as score_file:
            score = score_file.read()
        posts = list(filter(None, re.split("[USER\d+]", thread)))
        post_ids = thread_to_post[threadid]
        for post, post_id in zip(posts, post_ids):
            if post_id not in seen_ids and is_component_in_post(post, claims, premises):
                with open(post_dir + "/" + post_id + ".txt", "w") as post_content:
                    post_content.write(post)
                with open(post_dir + "/" + post_id + ".claim", "w") as _claim:
                    _claim.write(claims)
                with open(post_dir + "/" + post_id + ".premise", "w") as _premise:
                    _premise.write(premises)
                with open(post_dir + "/" + post_id + ".score", "w") as _premise:
                    _premise.write(score)
                post_annotated += 1
    print("{} annotated posts added.".format(post_annotated))
