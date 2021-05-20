import glob
import sys
import tensorflow as tf

from configs import config, tokenizer
from tokenize_components import get_model_inputs
from simp_utils import labels_to_tags

def generator(file_list):
    for elem in get_model_inputs(file_list):
        yield elem

def get_datasets(file_list):

     def callable_gen():
         nonlocal file_list
         for elem in generator(file_list):
             yield elem

     return tf.data.Dataset.from_generator(callable_gen,
                                           output_signature=(tf.TensorSpec(shape=(), dtype=tf.string, name='filename'),
                                                             tf.TensorSpec(shape=(None), dtype=tf.int32, name='tokenized_thread'),
                                                             tf.TensorSpec(shape=(None), dtype=tf.int32, name='comp_type_labels'),
                                                             tf.TensorSpec(shape=(None, 3), dtype=tf.int32, name='refers_to_and_type'),
                                                             tf.TensorSpec(shape=(None), dtype=tf.int32, name='attention_mask'),
                                                             tf.TensorSpec(shape=(None), dtype=tf.int32, name='global_attention_mask'))
                                          ).padded_batch(config['batch_size'],
                                                         padded_shapes=([],[None],[None],[None, None],[None],[None]),
                                                         padding_values=(None, *tuple(config['pad_for'].values())),
                                                         ).cache()

with open('./train_test_split.txt') as f:
    lines = f.read()
    train_files = [elem.split('/')[-1][:-4].split('_') for elem in list(filter(None, lines.split('\n')[0].split('\t')))]
    test_files = [elem.split('/')[-1][:-4].split('_') for elem in list(filter(None, lines.split('\n')[1].split('\t')))]
    train_files = ['./AmpersandData/change-my-view-modes/v2.0/'+elem[1]+'/'+elem[0]+'.xml' for elem in train_files]
    test_files = ['./AmpersandData/change-my-view-modes/v2.0/'+elem[1]+'/'+elem[0]+'.xml' for elem in test_files]
train_dataset = get_datasets(train_files)
test_dataset = get_datasets(test_files)

"""## Loading Model"""

from my_utils import get_model

model = get_model(config['max_tokenizer_length'], config['attention_window'])
model.resize_token_embeddings(len(tokenizer))

from models.TaskModel import TaskModel
from transformers import create_optimizer

task_model = TaskModel(model.layers[0],
                  max_trans=0.1, min_trans=-0.1,
                  is_padded=False)

task_optimizer, _ = create_optimizer(init_lr = 0.00005,
                             num_train_steps = 800,
                             num_warmup_steps = 80)
task_ckpt = tf.train.Checkpoint(task_model=task_model, task_optimizer=task_optimizer)
task_ckpt_manager = tf.train.CheckpointManager(task_ckpt, '../SavedModels/LF_With_rel_preds', max_to_keep=40)

"""## Train step"""

#@tf.function(input_signature=[(tf.TensorSpec(shape=(None, None), dtype=tf.int32, name='tokenized_thread'),
#                               tf.TensorSpec(shape=(None, None), dtype=tf.int32, name='comp_type_labels'),
#                               tf.TensorSpec(shape=(None, 3), dtype=tf.int32, name='refers_to_and_type'),
#                               tf.TensorSpec(shape=(None, None), dtype=tf.int32, name='attention_mask'),
#                               tf.TensorSpec(shape=(None, None), dtype=tf.int32, name='global_attention_mask'))])
def batch_train_step(inp):
    tokenized_thread, comp_type_labels, refers_to_and_type, attention_mask, global_attention_mask = inp
    inputs = {}
    inputs['input_ids'] = tokenized_thread
    inputs['attention_mask'] = attention_mask
    inputs['global_attention_mask'] = global_attention_mask
    with tf.GradientTape() as tape:
        crf_loss, cc_loss, relations_loss  = task_model.compute_loss(inputs, (comp_type_labels, refers_to_and_type))
        tf.print("Losses: ", crf_loss, cc_loss, relations_loss, output_stream=sys.stdout)
        total_loss = crf_loss + cc_loss + relations_loss

    gradients = tape.gradient(total_loss, task_model.trainable_variables)
    task_optimizer.apply_gradients(zip(gradients, task_model.trainable_variables))

#for elem in train_dataset:
#    print("Sample iteration of train dataset: ", elem[0], elem[1])
#    print("Trying training..")
#    batch_train_step(elem[1:])
#    break

"""## Eval Step"""

#@tf.function(input_signature=[(tf.TensorSpec(shape=(None, None), dtype=tf.int32, name='tokenized_thread'),
#                               tf.TensorSpec(shape=(None, None), dtype=tf.int32, name='comp_type_labels'),
#                               tf.TensorSpec(shape=(None, 3), dtype=tf.int32, name='refers_to_and_type'),
#                               tf.TensorSpec(shape=(None, None), dtype=tf.int32, name='attention_mask'),
#                               tf.TensorSpec(shape=(None, None), dtype=tf.int32, name='global_attention_mask'))])
def batch_eval_step(inp):
    tokenized_thread, comp_type_labels, refers_to_and_type, attention_mask, global_attention_mask = inp
    inputs = {}
    inputs['input_ids'] = tokenized_thread
    inputs['attention_mask'] = attention_mask
    inputs['global_attention_mask'] = global_attention_mask
    viterbi_seqs, seq_lens, optimal_trees = task_model.infer_step(inputs)
    return viterbi_seqs, seq_lens, optimal_trees

"""## Training Loop"""

#result_file = open('../Results/LF_with_rel_preds.txt', 'w')
#pbar = tf.keras.utils.Progbar(target=800, 
#                              unit_name='step')
steps = 0
ix = 1
train_iter = iter(train_dataset.repeat())

print("Starting Training..")
    inp = train_iter.get_next()
while steps<800:
    batch_train_step(inp[1:])
    steps += 1
    print("Step: ", steps)
#    pbar.add(1)

"""## Evaluation Loop"""

from seqeval.metrics import classification_report

from evaluate_relation_preds import single_sample_eval

L, P = [], []
for inp in test_dataset:
    viterbi_seqs, seq_lens, optimal_trees = batch_eval_step(inp[1:])
    #print(viterbi_seqs, seq_lens, relation_type_preds, refers_preds)
    #print(viterbi_seqs[0][:seq_lens[0]])
    single_sample_eval(inp[0], seq_lens, viterbi_seqs, optimal_trees)
    
    for p, l, length in zip(list(viterbi_seqs.numpy()), list(inp[2].numpy()), list(seq_lens.numpy())):
        true_tag = labels_to_tags(l)
        predicted_tag = labels_to_tags(p)
        L.append(true_tag[:length])
        P.append(predicted_tag[:length])

s = classification_report(L, P)
print("Classfication Report: ", s)
