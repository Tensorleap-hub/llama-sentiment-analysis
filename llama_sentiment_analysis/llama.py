from typing import List
import numpy as np
import tensorflow as tf
from transformers import AutoTokenizer
import os
from code_loader.inner_leap_binder.leapbinder_decorators import tensorleap_custom_loss

from llama_sentiment_analysis.config import CONFIG
from llama_sentiment_analysis.utils.tokenizer_utils import tokenize_sentence

dir_path = os.path.dirname(os.path.abspath(__file__))
tokenizer_path = os.path.join(dir_path, 'model')
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
tokenizer.pad_token = tokenizer.eos_token


def tokenize_and_align_labels(examples):
    """ Given batch of examples, tokenize the samples and align the labels
     accordingly """
    idx, label, sentence = examples["idx"], examples["label"], examples["sentence"]

    tokenized_inputs = tokenize_sentence(tokenizer, sentence[0])
    tokenized_inputs["labels"] = label
    return tokenized_inputs

def get_label_from_prediction(attention_mask, prediction):
    last_token_index = np.where(attention_mask[0] == 1)[0][-1]
    pred_token_logits = prediction[:, last_token_index, :]
    pred_token_text = tokenizer.convert_ids_to_tokens(int(np.argmax(pred_token_logits[0])))
    return pred_token_logits, pred_token_text, last_token_index

def get_labels_ids_map():
    labels_to_ids_dict = {}
    for label in CONFIG["labels"]:
        labels_to_ids_dict[label] = tokenizer.convert_tokens_to_ids(label)
    return labels_to_ids_dict

def get_vocab_size():
    return tokenizer.vocab_size

@tensorleap_custom_loss(name="CE_loss")
def CE_loss(ground_truth: np.ndarray, attention_masks: np.ndarray, preds: np.ndarray) -> np.ndarray:
    prediction, pred_token_text, last_token_index = get_label_from_prediction(attention_masks, preds)
    prediction = prediction[0]  # TODO: this is not the same locally but same on docker, why?
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction='none')
    loss_val = loss(np.expand_dims(tf.one_hot(int(ground_truth[0]), depth=len(prediction)), axis=0), np.expand_dims(prediction, axis=0))
    return np.expand_dims([tf.reduce_mean(loss_val).numpy()], axis=0)

