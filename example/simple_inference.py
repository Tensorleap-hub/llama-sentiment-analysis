from transformers import AutoTokenizer
import numpy as np
import tensorflow as tf
from llama_sentiment_analysis.utils.tokenizer_utils import tokenize_sentence

def simple_inference():
    # ----------------------------------------- Input Preparation --------------------------
    tokenizer = AutoTokenizer.from_pretrained("../model")
    tokenizer.pad_token = tokenizer.eos_token
    text = "i love this movie!"

    keras_inputs = tokenize_sentence(tokenizer, text)

    model = tf.keras.models.load_model('../model/llama_32_1b_inst.h5')
    # --------------------------------- Evaluating Inference -------------------------------------
    outputs = model(keras_inputs)
    last_token_logits = outputs[0, -1]
    pred_token_id = np.argmax(last_token_logits)
    pred_token = tokenizer.decode([pred_token_id]).strip().lower()
    print(f'sentence: {text}, prediction: {pred_token}')

if __name__ == '__main__':
    simple_inference()