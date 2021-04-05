import os
import random
import numpy as np
import sys
import json
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from transformers import LongformerTokenizer, TFLongformerModel
from utils import get_custom_model, labels_to_tags
from TaskModel import TaskModel
from LF_data_pipeline import make_dataset
from seqeval.metrics import classification_report
from params import ContextModelParams as MParams

class ContextModelMain:
    def __init__(self, checkpoint_path):
        self.checkpoint_path = checkpoint_path
        self.tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
        with open(MParams['sp_tokens']) as f:
            self.tokenizer.add_tokens(list(filter(None, f.read().split('\n'))))

    def load_model(self, learning_rate=5e-5):
        encoder = TFLongformerModel.from_pretrained('allenai/longformer-base-4096')
        encoder.resize_token_embeddings(len(self.tokenizer))
        model = TaskModel(encoder,
                          max_trans=0.1,
                          min_trans=-0.1,
                          is_padded=True)
        return model

    def train_model(self, data_dir, nb_epochs):
        model = self.load_model()
        ckpt = tf.train.Checkpoint(model=model)
        ckpt_manager = tf.train.CheckpointManager(ckpt, self.checkpoint_path, max_to_keep=8)
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
            print('Latest checkpoint restored; Model was trained for {} steps.'.format(ckpt.model.optimizer.iterations.numpy()))
        else:
            print('ContextModel initialized from scratch!')
        train_dataset = make_dataset(data_dir, self.tokenizer,
                                     is_labeled=True, batch_size=7936)
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
        for epoch in range(1, nb_epochs+1):
            print('\nTraining epoch {}'.format(epoch))
            pbar = tf.keras.utils.Progbar(target=len(data_dir), 
                                  unit_name='sample')
            for inp in train_dataset:
                batch_train_step(inp)
                pbar.add(inp[1].numpy().shape[0])
        ckpt_save_path = ckpt_manager.save()

    def evaluate_model(self, data_dir):
        model = self.load_model()
        ckpt = tf.train.Checkpoint(model=model)
        ckpt_manager = tf.train.CheckpointManager(ckpt, self.checkpoint_path, max_to_keep=8)
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
            print('Latest checkpoint restored; Model was trained for {} steps.'.format(ckpt.model.optimizer.iterations.numpy()))
        else:
            print('ContextModel initialized from scratch!')
        test_dataset = make_dataset(data_dir, self.tokenizer,
                                     is_labeled=True, batch_size=7936)
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
            predictions, _ = model.infer_step(inputs)
            return predictions, labels
        print('\n Testing...')
        pbar = tf.keras.utils.Progbar(target=len(data_dir), 
                                      unit_name='sample')
        L, P = [], []
        for inp in test_dataset:
            predictions,labels = batch_evaluate_step(inp)
            pbar.add(inp[1].numpy().shape[0])
            for p, l in zip(list(predictions.numpy()), list(labels.numpy())):
                true_tag = labels_to_tags(l)
                predicted_tag = labels_to_tags(p)
                L.append(true_tag)
                P.append(predicted_tag)
        print(classification_report(L, P))

    def annotate(self, data_dir, save_dir, num_files):
        model = self.load_model()
        ckpt = tf.train.Checkpoint(model=model)
        ckpt_manager = tf.train.CheckpointManager(ckpt, self.checkpoint_path, max_to_keep=8)
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
            print('Latest checkpoint restored; Model was trained for {} steps.'.format(ckpt.model.optimizer.iterations.numpy()))
        else:
            print('ContextModel initialized from scratch!')
        dataset = make_dataset(data_dir, self.tokenizer,
                               is_labeled=False, batch_size=8448)
        @tf.function(input_signature=[(tf.TensorSpec([None], tf.string, name="idx"),
                               tf.TensorSpec([None, None], tf.int32, name="input_ids"),
                               tf.TensorSpec([None, None], tf.int32, name="attention_mask"),
                               tf.TensorSpec([None, None], tf.int32, name="global_attention_mask"))])

        def batch_predict_step(inp):
            idx, input_ids, attention_mask, global_attention_mask = inp
            inputs = {}
            inputs['input_ids'] = input_ids
            inputs['attention_mask'] = attention_mask
            inputs['global_attention_mask'] = global_attention_mask
            predictions, scores = model.infer_step(inputs)
            return idx, scores, input_ids, predictions

        def get_segment(encoded_text, tag):
            '''Extract string components from text using predicted tags'''
            claims, premises = [], []
            cseg, pseg = [], []
            for token, tag in zip(list(encoded_text), list(tag)):
                if tag == 1:
                    '''Start of claim; push previous segments (if any), and initialize claim array; '''
                    if len(pseg):
                        premises.append(self.tokenizer.decode(pseg))
                        pseg = []
                    if len(cseg):
                        claims.append(self.tokenizer.decode(cseg))
                    cseg = [token]
                elif tag == 2:
                    '''Continue claim; push previous premise (if any), append token to claim (if any)'''
                    if len(pseg):
                        premises.append(self.tokenizer.decode(pseg))
                        pseg = []
                    if len(cseg):
                        cseg.append(token)
                elif tag == 3:
                    '''Start of premise; push previous segments (if any), and initialize premise array; '''
                    if len(cseg):
                        claims.append(self.tokenizer.decode(cseg))
                        cseg = []
                    if len(pseg):
                        premises.append(self.tokenizer.decode(pseg))
                    pseg = [token]
                elif tag == 4:
                    '''Continue premise; push previous claim (if any), append token to premise (if any)'''
                    if len(cseg):
                        claims.append(self.tokenizer.decode(cseg))
                        cseg = []
                    if len(pseg):
                        pseg.append(token)
                else:
                    '''other or pad token; push all previous segments (if any)'''
                    if len(pseg):
                        premises.append(self.tokenizer.decode(pseg))
                        pseg = []
                    if len(cseg):
                        claims.append(self.tokenizer.decode(cseg))
                        cseg = []
            return claims, premises
        IDs, Scores, Encoded_inputs, Tags = [], [], [], []
        pbar = tf.keras.utils.Progbar(target=num_files, 
                                      unit_name='sample')
        for inp in dataset:
            idx, scores, input_ids, tags = batch_predict_step(inp)
            IDs.extend(list(idx.numpy()))
            Scores.extend(list(scores.numpy()))
            Encoded_inputs.extend(list(input_ids.numpy()))
            Tags.extend(list(tags.numpy()))
            pbar.add(inp[1].numpy().shape[0])
        for idx, s, encoded_text, tag in zip(IDs, Scores, Encoded_inputs, Tags):
            '''Write claims and premises to files'''
            claims, premises = get_segment(encoded_text, tag)
            with open(save_dir+'/'+idx.decode('utf-8')+'.claim', 'w') as f:
                f.write('\n'.join(claims))
            with open(save_dir+'/'+idx.decode('utf-8')+'.premise', 'w') as f:
                f.write('\n'.join(premises))
            with open(save_dir+'/'+idx.decode('utf-8')+'.score', 'w') as f:
                f.write(str(s))

