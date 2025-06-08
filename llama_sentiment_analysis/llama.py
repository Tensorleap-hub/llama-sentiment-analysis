from typing import List
import numpy as np
import tensorflow as tf
from transformers import AutoTokenizer

from code_loader.inner_leap_binder.leapbinder_decorators import tensorleap_custom_loss

from llama_sentiment_analysis.config import CONFIG
from llama_sentiment_analysis.utils.tokenizer_utils import tokenize_sentence

model_label2id = {"positive": 1,
                    "negative": 0
                    }
model_id2label = {v: k for k, v in model_label2id.items()}

map_idx_to_label = dict(enumerate(CONFIG["labels"]))
# map_label_idx_to_cat = {i: c.split("-")[-1] for i, c in map_idx_to_label.items()}

model_labels_to_ds_label = {0: "negative", 1: "positive"}

tokenizer = AutoTokenizer.from_pretrained("model")
tokenizer.pad_token = tokenizer.eos_token

def map_model_to_ds_labels(predicted_labels):
    labels_lst = CONFIG["labels"]
    # int labels to category names
    mapped_labels = [[model_id2label[i] for i in labels] for labels in predicted_labels]
    # category names to ds int labels
    mapped_labels = [[labels_lst[i] for i in labels] for labels in mapped_labels]
    return mapped_labels


def tokenize_and_align_labels(examples):
    """ Given batch of examples, tokenize the samples and align the labels
     accordingly """
    idx, label, sentence = examples["idx"], examples["label"], examples["sentence"]

    tokenized_inputs = tokenize_sentence(tokenizer, sentence[0])
    tokenized_inputs["labels"] = label
    return tokenized_inputs

def get_labels_ids_map():
    labels_to_ids_dict = {}
    for label in CONFIG["labels"]:
        labels_to_ids_dict[label] = tokenizer.convert_tokens_to_ids(label)
    return labels_to_ids_dict

def get_vocab_size():
    return tokenizer.vocab_size

def postprocess_predictions(predictions: tf.Tensor, input_ids: List[int] = None):
    """ given predictions tensor return a list of the labels
    if gt labels are given mask based on -100 token """
    # Classes
    label_names = CONFIG["labels"]

    # Logits to predicted labels
    predictions = transform_prediction(predictions)
    predictions = predictions.numpy()
    # Take argmax as the index label
    predictions = predictions.argmax(-1)
    if input_ids is not None:
        CLS_ID, SEP_ID = CONFIG["CLS_ID"], CONFIG["SEP_ID"]
        # Find the positions of `[CLS]` and `[SEP]`
        cls_positions = [np.where(sublist == CLS_ID)[0] for sublist in input_ids]
        sep_positions = [np.where(sublist == SEP_ID)[0] for sublist in input_ids]

        # Use the first `[CLS]` and the last `[SEP]`, if not valid then ignore and map all
        starts = [cls_pos[0] if cls_pos.size > 0 else 0 for cls_pos in cls_positions]
        ends = [sep_pos[-1] if sep_pos.size > 0 else len(predictions[0]) for sep_pos in sep_positions]

        # Convert to labels based on non masked tokens
        true_predictions = [
            [label_names[int(p)] if starts[j] < i < ends[j] else "" for i, p in enumerate(prediction)]
            for j, prediction in enumerate(predictions)
        ]
    else:
        true_predictions = [
            [label_names[int(p)] for p in prediction]
            for prediction in predictions]
    return true_predictions


def postprocess_labels(labels: List[int]):
    # Classes
    label_names = CONFIG["labels"]
    # Remove ignored index (special tokens) and convert to labels
    true_labels = [[label_names[l] if l != -100 else "" for l in label] for label in labels]
    return true_labels


def decode_token_ids(input_ids: List[int]) -> List[str]:
    text_tokens = tokenizer.convert_ids_to_tokens(input_ids)
    # Clean the text tokens
    text_tokens = [token.replace("##", "") if token not in ['[CLS]', '[SEP]', '[PAD]'] else '' for token in text_tokens]
    return text_tokens


def decode(tensor):
    "Given logits tensor return the labels classifying each token "
    idx_outputs, labels_outputs = [], []
    tensor = tf.expand_dims(tensor, 0) if len(tensor.shape) == 2 else tensor
    for i in range(tensor.shape[0]):
        pred_idx = np.argmax(tensor[i].numpy(), -1)
        labels = [map_idx_to_label[i] for i in pred_idx]
        # Truncate/ Pad the tensors
        pred_idx, labels = truncate_pad(pred_idx), truncate_pad(labels)
        idx_outputs.append(pred_idx)
        labels_outputs.append(labels)
    return idx_outputs, labels_outputs


def truncate_pad(decoded: List[str], token=0) ->List[str]:
    """
    Description: Truncates or pads the decoded tokens to match the maximum sequence length.
    Parameters:
    decoded (List[str]): List of decoded tokens.
    Returns:
    decoded (List[str]): List of truncated or padded tokens.
    """
    max_length = CONFIG['max_length']
    if len(decoded) < max_length:  # pad
        decoded += (max_length - len(decoded)) * [token]
    elif len(decoded) > max_length:  # truncate
        decoded = decoded[:max_length]
    return decoded

@tensorleap_custom_loss(name="CE_loss")
def CE_loss(ground_truth: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction='none')
    # -100 label encoded as zero vec i.e., (0, 0, .., 0) thus we don't need to mask in the loss
    loss_val = loss(tf.one_hot(ground_truth, depth=len(prediction)).numpy(), np.expand_dims(prediction.numpy(), axis=0))
    # loss_val = tf.reduce_mean(loss_val, axis=1)
    return loss_val.numpy()


def mask_one_hot_labels(ground_truth):
    """ Given GT one hot encoded mask return bool mask for valid tokens i.e., exclude CLS SEP and PAD tokens """
    ground_truth = tf.reduce_sum(ground_truth, -1)
    mask = tf.math.not_equal(ground_truth, 0)       # why 0?
    return mask


def mask_based_inputs(input_ids):
    """ Given input ids return bool mask for valid tokens
    That is until begin of 0 token ids """
    mask = ~tf.equal(input_ids, 0)
    return mask

