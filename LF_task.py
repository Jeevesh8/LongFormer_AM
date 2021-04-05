import random
import numpy as np
import sys
import json
import tensorflow as tf
from transformers import LongformerTokenizer, TFLongformerForMaskedLM, create_optimizer
from utils import get_custom_model, labels_to_tags
from TaskModel import TaskModel
from LF_data_pipeline import make_dataset
from seqeval.metrics import classification_report
'''
model, tokenizer = get_custom_model(8192, 192)
'''
tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
'''
model = TFLongformerForMaskedLM.from_pretrained('allenai/longformer-base-4096')
'''
with open('special_tokens.txt') as f:
    sp_tokens = list(filter(None, f.read().split('\n')))
tokenizer.add_tokens(sp_tokens)
'''
model.resize_token_embeddings(len(tokenizer))
optimizer, _ = create_optimizer(init_lr = 0.0001,
                             num_train_steps = 200000,
                             num_warmup_steps = 10000)

ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
'''
'''
ckpt_manager = tf.train.CheckpointManager(ckpt, '../SavedModels/finetunedLongformer', max_to_keep=20)
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
    print('Latest checkpoint restored; Model was trained for {} steps.'.format(ckpt.optimizer.iterations.numpy()))
else:
    print('Model initialized from scratch!')
'''
'''
ckpt.restore('../SavedModels/finetunedLongformer/ckpt-11').expect_partial()
print('Checkpoint restored; Model was trained for {} steps.'.format(ckpt.optimizer.iterations.numpy()))
task_model = TaskModel(model.layers[0],
                  max_trans=0.1, min_trans=-0.1,
                  use_dsc=False,
                  is_padded=True)
task_optimizer, _ = create_optimizer(init_lr = 0.00005,
                             num_train_steps = 800,
                             num_warmup_steps = 80)
task_ckpt = tf.train.Checkpoint(task_model=task_model, task_optimizer=task_optimizer)
task_ckpt_manager = tf.train.CheckpointManager(task_ckpt, '../SavedModels/DiaFormer-11', max_to_keep=40)
'''
with open('../Data/train_test_split.txt') as f:
    lines = f.read()
    train_files = list(filter(None, lines.split('\n')[0].split('\t')))
    test_files = list(filter(None, lines.split('\n')[1].split('\t')))[10:]

train_dataset = make_dataset(train_files, tokenizer, is_labeled=True, maxlen=4096, batch_size=8194-1024)
test_dataset = make_dataset(test_files, tokenizer, is_labeled=True, maxlen=4096, batch_size=8194)
count_claims, count_premises = 0, 0
for inp in train_dataset:
    labels = inp[-1]
    count_claims += tf.reduce_sum(tf.cast(tf.equal(labels, 1), dtype=tf.int32))
    count_premises += tf.reduce_sum(tf.cast(tf.equal(labels, 3), dtype=tf.int32))
print('In train data:', count_claims, count_premises)
count_claims, count_premises = 0, 0
for inp in test_dataset:
    labels = inp[-1]
    count_claims += tf.reduce_sum(tf.cast(tf.equal(labels, 1), dtype=tf.int32))
    count_premises += tf.reduce_sum(tf.cast(tf.equal(labels, 3), dtype=tf.int32))
print('In test data:', count_claims, count_premises)

'''
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
    sample_weight = task_model.compute_batch_sample_weight(labels)
    with tf.GradientTape() as tape:
        crf_loss, cc_loss  = task_model.compute_loss(inputs, labels, sample_weight)
        total_loss = crf_loss + cc_loss
    gradients = tape.gradient(total_loss, task_model.trainable_variables)
    task_optimizer.apply_gradients(zip(gradients, task_model.trainable_variables))

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
    predictions, seq_lens = task_model.infer_step(inputs)
    return predictions, labels, seq_lens

result_file = open('../Results/DiaFormer_800_80_11.txt', 'w')
pbar = tf.keras.utils.Progbar(target=800, 
                              unit_name='step')
steps = 0
ix = 1
train_iter = iter(train_dataset.repeat())
while steps<800:
    inp = train_iter.get_next()
    batch_train_step(inp)
    steps += 1
    pbar.add(1)
    if steps%25 == 0:
        L, P = [], []
        for inp in test_dataset:
            predictions, labels, s_lens = batch_evaluate_step(inp)
            for p, l, length in zip(list(predictions.numpy()), list(labels.numpy()), list(s_lens.numpy())):
                true_tag = labels_to_tags(l)
                predicted_tag = labels_to_tags(p)
                L.append(true_tag[:length])
                P.append(predicted_tag[:length])
        s = classification_report(L, P)
        result_file.write('Checkpoint index {}'.format(ix))
        result_file.write(s)
        _ = task_ckpt_manager.save()
        ix += 1
result_file.close()
'''
