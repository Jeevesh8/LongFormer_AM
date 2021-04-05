import tensorflow as tf
from transformers import LongformerTokenizer, TFLongformerForMaskedLM, create_optimizer
from finetune_pipeline import make_dataset
from utils import get_custom_model
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def mlm_loss(logits, labels, mask_value):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    weights = tf.cast(tf.equal(labels, mask_value), tf.float32)
    return tf.reduce_sum(xentropy * weights, axis=-1) / (tf.reduce_sum(weights, axis=-1)+tf.keras.backend.epsilon())
model, tokenizer = get_custom_model(8192, 192)
with open('special_tokens.txt') as f:
    sp_tokens = list(filter(None, f.read().split('\n')))
tokenizer.add_tokens(sp_tokens)
model.resize_token_embeddings(len(tokenizer))
optimizer, _ = create_optimizer(init_lr = 0.0001,
                             num_train_steps = 200000,
                             num_warmup_steps = 10000)

ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, '../SavedModels/finetunedLongformer-8192', max_to_keep=20)
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print('Latest checkpoint restored; Model was trained for {} steps.'.format(optimizer.iterations.numpy()))
else:
    print('Model initialized from scratch!')

signature = (tf.TensorSpec([None, None], tf.int32, name="input_ids"),
             tf.TensorSpec([None, None], tf.int32, name="attention_mask"),
             tf.TensorSpec([None, None], tf.int32, name="global_attention_mask"),
             tf.TensorSpec([None, None], tf.int32, name="mlm_labels"))
@tf.function(input_signature=[signature])
def train_fn(inp):
    inputs = {}
    inputs['input_ids'] = inp[0]
    inputs['attention_mask'] = inp[1]
    inputs['global_attention_mask'] = inp[2]

    with tf.GradientTape() as tape:
        logits = model(inputs, training=True)
        loss = mlm_loss(logits['logits'], inp[3], tokenizer.mask_token_id)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

train_data = make_dataset(tokenizer, '../Data/RawThreads', 8194, maxlen=8192)
pbar = tf.keras.utils.Progbar(target=200000,
                              width=15, 
                              interval=0.005)
steps = 0
for _ in range(2):
    for inputs in train_data:
        train_fn(inputs)
        pbar.add(1)
        steps +=1
        if steps%10000 == 0:
            path = ckpt_manager.save()
    print('\nEpoch completed')
