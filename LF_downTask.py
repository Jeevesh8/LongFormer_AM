import random
import numpy as np
import sys
import json
import tensorflow as tf
from transformers import LongformerTokenizer
from utils import get_custom_model, labels_to_tags
from TaskModel import TaskModel
from LF_data_pipeline import make_dataset
from seqeval.metrics import classification_report

tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
with open('special_tokens.txt') as f:
    tokenizer.add_tokens(list(filter(None, f.read().split('\n'))))
model = TaskModel(get_custom_model(len(tokenizer), 'LF'),
                  max_trans=0.1, min_trans=-0.1,
                  is_padded=True)

with open('../Data/train_test_split.txt') as f:
    lines = f.read()
    train_files = list(filter(None, lines.split('\n')[0].split('\t')))
    test_files = list(filter(None, lines.split('\n')[1].split('\t')))

train_dataset = make_dataset(train_files, tokenizer, is_labeled=True, batch_size=7936)
test_dataset = make_dataset(test_files, tokenizer, is_labeled=True, batch_size=7936)

@tf.function(input_signature=[(tf.TensorSpec([None], tf.string, name="idx"),
                               tf.TensorSpec([None, None], tf.int32, name="input_ids"), 
                               tf.TensorSpec([None, None], tf.int32, name="attention_mask"),
                               tf.TensorSpec([None, None], tf.int32, name="global_attention_mask"),
                               tf.TensorSpec([None, None], tf.int32, name="labels"))])
def batch_train_step(inp):
    idx, input_ids, attention_mask, global_attention_mask, labels = inp
    inputs = {}
    inputs['input_ids'] = input_ids
    inputs['attention_mask'] = attention_mask
    inputs['global_attention_mask'] = global_attention_mask
    sample_weight = model.compute_batch_sample_weight(labels)
    with tf.GradientTape() as tape:
        crf_loss, cc_loss  = model.compute_loss(inputs, labels, sample_weight)
        total_loss = crf_loss + cc_loss
    gradients = tape.gradient(total_loss, model.trainable_variables)
    model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

@tf.function(input_signature=[(tf.TensorSpec([None], tf.string, name="idx"),
                               tf.TensorSpec([None, None], tf.int32, name="input_ids"), 
                               tf.TensorSpec([None, None], tf.int32, name="attention_mask"),
                               tf.TensorSpec([None, None], tf.int32, name="global_attention_mask"),
                               tf.TensorSpec([None, None], tf.int32, name="labels"))])
def batch_evaluate_step(inp):
    idx, input_ids, attention_mask, global_attention_mask, labels = inp
    inputs = {}
    inputs['input_ids'] = input_ids
    inputs['attention_mask'] = attention_mask
    inputs['global_attention_mask'] = global_attention_mask
    predictions = model.infer_step(inputs)
    return predictions

result_file = open('../Results/Longformer_results_withdsc.txt', 'w')
for epoch in range(1, 21):
    print('\nTraining epoch {}'.format(epoch))
    pbar = tf.keras.utils.Progbar(target=len(train_files), 
                                  unit_name='sample')
    for inp in train_dataset:
        batch_train_step(inp)
        pbar.add(inp[1].numpy().shape[0])
    if epoch>=1:
        print('\n Testing...')
        pbar = tf.keras.utils.Progbar(target=len(test_files), 
                                      unit_name='sample')
        L, P = [], []
        for inp in test_dataset:
            predictions = batch_evaluate_step(inputs)
            pbar.add(inp[1].numpy().shape[0])
            for p, l in zip(list(predictions.numpy()), list(labels.numpy())):
                true_tag = labels_to_tags(l)
                predicted_tag = labels_to_tags(p)
                L.append(true_tag)
                P.append(predicted_tag)
        result_file.write(classification_report(L, P))
result_file.close()
