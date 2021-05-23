import argparse
import glob
import sys
import tensorflow as tf
from transformers import create_optimizer
from seqeval.metrics import classification_report

from configs import config, tokenizer
from tokenize_components import get_model_inputs
from simp_utils import labels_to_tags
from my_utils import get_model
from models.TaskModel import TaskModel
from evaluate_relation_preds import single_sample_eval

"""## Loading Model"""

model = get_model(config["max_tokenizer_length"], config["attention_window"])
model.resize_token_embeddings(len(tokenizer))

task_model = TaskModel(model.layers[0], max_trans=0.1, min_trans=-0.1, is_padded=False)

task_optimizer, _ = create_optimizer(
    init_lr=0.00005, num_train_steps=800, num_warmup_steps=80
)

task_ckpt = tf.train.Checkpoint(task_model=task_model, task_optimizer=task_optimizer)
task_ckpt_manager = tf.train.CheckpointManager(
    task_ckpt, "../SavedModels/LF_With_rel_preds", max_to_keep=40
)


def generator(file_list):
    for elem in get_model_inputs(file_list):
        yield elem


def get_datasets(file_list):
    def callable_gen():
        nonlocal file_list
        for elem in generator(file_list):
            yield elem

    return (
        tf.data.Dataset.from_generator(
            callable_gen,
            output_signature=(
                tf.TensorSpec(shape=(), dtype=tf.string, name="filename"),
                tf.TensorSpec(shape=(None), dtype=tf.int32, name="tokenized_thread"),
                tf.TensorSpec(shape=(None), dtype=tf.int32, name="comp_type_labels"),
                tf.TensorSpec(shape=(None, None), dtype=tf.int32, name="refers_labels"),
                tf.TensorSpec(
                    shape=(None), dtype=tf.int32, name="relation_type_labels"
                ),
                tf.TensorSpec(shape=(None), dtype=tf.int32, name="attention_mask"),
                tf.TensorSpec(
                    shape=(None), dtype=tf.int32, name="global_attention_mask"
                ),
            ),
        )
        .padded_batch(
            config["batch_size"],
            padded_shapes=([], [None], [None], [None, None], [None], [None], [None]),
            padding_values=(None, *tuple(config["pad_for"].values())),
        )
        .cache()
    )


def get_train_test_data(train_sz, test_sz, op_wise_split):
    with open(op_wise_split) as f:
        lines = f.readlines()

    test_files, train_files = [], []

    for threads in lines:
        files = [
            elem.strip("\"'") for elem in threads.strip("\n").strip("[]").split(", ")
        ]
        if len(test_files) < test_sz:
            test_files += files
        elif len(train_files) < train_sz:
            train_files += files

    train_dataset = get_datasets(train_files)
    test_dataset = get_datasets(test_files)

    return train_dataset, test_dataset


"""## Train step"""


@tf.function(
    input_signature=[
        (
            tf.TensorSpec(shape=(None, None), dtype=tf.int32, name="tokenized_thread"),
            tf.TensorSpec(shape=(None, None), dtype=tf.int32, name="comp_type_labels"),
            tf.TensorSpec(
                shape=(None, None, None), dtype=tf.int32, name="refers_labels"
            ),
            tf.TensorSpec(
                shape=(None, None), dtype=tf.int32, name="relation_type_labels"
            ),
            tf.TensorSpec(shape=(None, None), dtype=tf.int32, name="attention_mask"),
            tf.TensorSpec(
                shape=(None, None), dtype=tf.int32, name="global_attention_mask"
            ),
        )
    ]
)
def batch_train_step(inp):
    (
        tokenized_thread,
        comp_type_labels,
        refers_labels,
        relation_type_labels,
        attention_mask,
        global_attention_mask,
    ) = inp
    inputs = {}
    inputs["input_ids"] = tokenized_thread
    inputs["attention_mask"] = attention_mask
    inputs["global_attention_mask"] = global_attention_mask
    with tf.GradientTape() as tape:
        (
            crf_loss,
            cc_loss,
            relation_type_cc_loss,
            refers_cc_loss,
        ) = task_model.compute_loss(
            inputs, (comp_type_labels, relation_type_labels, refers_labels)
        )
        tf.print(
            "Losses: ",
            crf_loss,
            cc_loss,
            relation_type_cc_loss,
            refers_cc_loss,
            output_stream=sys.stdout,
        )
        total_loss = crf_loss + cc_loss + relation_type_cc_loss + refers_cc_loss

    gradients = tape.gradient(total_loss, task_model.trainable_variables)
    task_optimizer.apply_gradients(zip(gradients, task_model.trainable_variables))


"""## Eval Step"""


@tf.function(
    input_signature=[
        (
            tf.TensorSpec(shape=(None, None), dtype=tf.int32, name="tokenized_thread"),
            tf.TensorSpec(shape=(None, None), dtype=tf.int32, name="comp_type_labels"),
            tf.TensorSpec(
                shape=(None, None, None), dtype=tf.int32, name="refers_labels"
            ),
            tf.TensorSpec(
                shape=(None, None), dtype=tf.int32, name="relation_type_labels"
            ),
            tf.TensorSpec(shape=(None, None), dtype=tf.int32, name="attention_mask"),
            tf.TensorSpec(
                shape=(None, None), dtype=tf.int32, name="global_attention_mask"
            ),
        )
    ]
)
def batch_eval_step(inp):
    (
        tokenized_thread,
        comp_type_labels,
        refers_labels,
        relation_type_labels,
        attention_mask,
        global_attention_mask,
    ) = inp
    inputs = {}
    inputs["input_ids"] = tokenized_thread
    inputs["attention_mask"] = attention_mask
    inputs["global_attention_mask"] = global_attention_mask
    viterbi_seqs, seq_lens, relation_type_preds, refers_preds = task_model.infer_step(
        inputs
    )
    return viterbi_seqs, seq_lens, relation_type_preds, refers_preds


"""## Training Loop"""


def train(train_dataset, test_dataset):
    steps = 0
    train_iter = iter(train_dataset.repeat())

    print("Starting Training..")
    while steps < 800:
        inp = train_iter.get_next()
        batch_train_step(inp[1:])
        steps += 1
        print("Step: ", steps)
        if steps % 50 == 0 or steps % 799 == 0:
            L, P = [], []
            for inp in test_dataset:
                (
                    viterbi_seqs,
                    seq_lens,
                    relation_type_preds,
                    refers_preds,
                ) = batch_eval_step(inp[1:])
                single_sample_eval(
                    inp[0], seq_lens, viterbi_seqs, refers_preds, relation_type_preds
                )
                for p, l, length in zip(
                    list(viterbi_seqs.numpy()),
                    list(inp[2].numpy()),
                    list(seq_lens.numpy()),
                ):
                    true_tag = labels_to_tags(l)
                    predicted_tag = labels_to_tags(p)
                    L.append(true_tag[:length])
                    P.append(predicted_tag[:length])
            s = classification_report(L, P)
            print("Classfication Report: ", s)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_sz",
        required=True,
        type=int,
        nargs="+",
        help="Number of threads to use for train set.",
    )
    parser.add_argument(
        "--test_sz", required=True, type=int, help="Number of threds in test set."
    )
    parser.add_argument(
        "--op_wise_split",
        default="./try_outs/op_wise_split.txt",
        help="path to file having op wise split of all threads.",
    )
    args = parser.parse_args()

    for train_sz in args.train_sz:
        train_dataset, test_dataset = get_train_test_data(
            train_sz, args.test_sz, args.op_wise_split
        )
        train(train_dataset, test_dataset)
