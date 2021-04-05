import random
import numpy as np
import sys
import json
import tensorflow as tf
from transformers import LongformerTokenizer, LongformerTokenizerFast, TFLongformerForMaskedLM, create_optimizer
from utils import get_custom_model, labels_to_tags
from TaskModel import TaskModel
from LF_data_pipeline import make_dataset
from seqeval.metrics import classification_report
tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
model = TFLongformerForMaskedLM.from_pretrained('allenai/longformer-base-4096')

with open('special_tokens.txt') as f:
    sp_tokens = list(filter(None, f.read().split('\n')))
tokenizer.add_tokens(sp_tokens)

with open('../ETC/markers.txt') as f:
    markers = list(filter(None, f.read().split('\n')))
markers = sorted(markers, key=lambda x: len(x.split()), reverse=True)
with open('../marker_tokens.txt', 'w') as f:
    for m in markers:
        enc_markers = tokenizer.encode(m)[1:-1]
        f.write('\t'.join([str(i) for i in enc_markers])+'\n')

raise ValueError

model.resize_token_embeddings(len(tokenizer))
task_model = TaskModel(model.layers[0],
                  max_trans=0.1, min_trans=-0.1,
                  use_dsc=False,
                  is_padded=True)
task_optimizer, _ = create_optimizer(init_lr = 0.00005,
                             num_train_steps = 500,
                             num_warmup_steps = 80)
task_ckpt = tf.train.Checkpoint(task_model=task_model, task_optimizer=task_optimizer)
task_ckpt.restore('../SavedModels/DiaFormer-11/ckpt-12')
print('Checkpoint restored; Model was trained for {} steps.'.format(task_ckpt.task_optimizer.iterations.numpy()))
with open('../Data/train_test_split.txt') as f:
    lines = f.read()
    train_files = list(filter(None, lines.split('\n')[0].split('\t')))
    test_files = list(filter(None, lines.split('\n')[1].split('\t')))[10:]

test_dataset = make_dataset(test_files, tokenizer, is_labeled=True, maxlen=4096, batch_size=8194)

@tf.function(input_signature=[(tf.TensorSpec([None], tf.string, name="idx"),
                               tf.TensorSpec([None, None], tf.int32, name="input_ids"), 
                               tf.TensorSpec([None, None], tf.int32, name="attention_mask"),
                               tf.TensorSpec([None, None], tf.int32, name="global_attention_mask"),
                               tf.TensorSpec([None, None], tf.int32, name="labels"))])
def batch_predict_step(inp):
    idx, input_ids, attention_mask, global_attention_mask, labels = inp
    inputs = {}
    inputs['input_ids'] = input_ids
    inputs['attention_mask'] = attention_mask
    inputs['global_attention_mask'] = global_attention_mask
    predictions, seq_lens = task_model.infer_step(inputs)
    #return idx, input_ids, predictions, seq_lens
    return labels, predictions, seq_lens

def get_segment(encoded_text, tag):
    '''Extract string components from text using predicted tags'''
    claims, premises = [], []
    cseg, pseg = [], []
    for token, tag in zip(list(encoded_text), list(tag)):
        if tag == 1:
            '''Start of claim; push previous segments (if any), and initialize claim array; '''
            if len(pseg):
                premises.append(tokenizer.decode(pseg))
                pseg = []
            if len(cseg):
                claims.append(tokenizer.decode(cseg))
            cseg = [token]
        elif tag == 2:
            '''Continue claim; push previous premise (if any), append token to claim (if any)'''
            if len(pseg):
                premises.append(tokenizer.decode(pseg))
                pseg = []
            if len(cseg):
                cseg.append(token)
        elif tag == 3:
            '''Start of premise; push previous segments (if any), and initialize premise array; '''
            if len(cseg):
                claims.append(tokenizer.decode(cseg))
                cseg = []
            if len(pseg):
                premises.append(tokenizer.decode(pseg))
            pseg = [token]
        elif tag == 4:
            '''Continue premise; push previous claim (if any), append token to premise (if any)'''
            if len(cseg):
                claims.append(tokenizer.decode(cseg))
                cseg = []
            if len(pseg):
                pseg.append(token)
        else:
            '''other or pad token; push all previous segments (if any)'''
            if len(pseg):
                premises.append(tokenizer.decode(pseg))
                pseg = []
            if len(cseg):
                claims.append(tokenizer.decode(cseg))
                cseg = []
    return claims, premises
'''
IDs, Encoded_inputs, Tags, Lens = [], [], [], []
for inp in test_dataset:
    idx, input_ids, tags, seq_lens = batch_predict_step(inp)
    IDs.extend(list(idx.numpy()))
    Encoded_inputs.extend(list(input_ids.numpy()))
    Tags.extend(list(tags.numpy()))
    Lens.extend(list(seq_lens.numpy()))
for idx, encoded_text, tag, l in zip(IDs, Encoded_inputs, Tags, Lens):
    claims, premises = get_segment(encoded_text[:l], tag[:l])
    file_id = idx.decode('utf-8').split('/')[-1].split('.')[0]
    with open('../Data/Predictions/'+file_id+'.claim', 'w') as f:
        f.write('\n'.join(claims))
    with open('../Data/Predictions/'+file_id+'.premise', 'w') as f:
        f.write('\n'.join(premises))
'''
X, y_true, y_test = [], [], []
for inp in test_dataset:
    input_ids = inp[1]
    labels, predictions, seq_lens = batch_predict_step(inp)
    for p, l, length, x in zip(list(predictions.numpy()), 
                            list(labels.numpy()), 
                            list(seq_lens.numpy()), 
                            list(input_ids.numpy())):
        true_tag = labels_to_tags(l)
        predicted_tag = labels_to_tags(p)
        y_true.extend(true_tag[:length])
        y_test.extend(predicted_tag[:length])
        X.extend(x[:length])
'''
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
tags = ['O', 'B-claim', 'I-claim', 'B-premise', 'I-premise']
cm = confusion_matrix(y_true, y_test, labels=tags)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=tags)
from matplotlib import pyplot as plt
fig, ax = plt.subplots()
disp.plot(ax=ax, cmap='Greys')
plt.savefig('../Results/Diaformer_ConfusionMatrix_greys.svg')
'''
idfile = open('../token_ids.txt', 'w')
gold_file = open('../gold_labels.txt', 'w')
pred_file = open('../pred_labels.txt', 'w')
idfile.write('\t'.join([str(x) for x in X]))
gold_file.write('\t'.join(y_true))
pred_file.write('\t'.join(y_test))
idfile.close()
gold_file.close()
pred_file.close()
