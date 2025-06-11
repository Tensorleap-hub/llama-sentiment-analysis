from transformers import AutoTokenizer
import numpy as np
import tensorflow as tf
from llama_sentiment_analysis.utils.tokenizer_utils import tokenize_sentence, tokenize_sentence_reg

def simple_inference_instruct():
    # ----------------------------------------- Input Preparation --------------------------
    tokenizer = AutoTokenizer.from_pretrained("../llama_sentiment_analysis/model")
    tokenizer.pad_token = tokenizer.eos_token
    text = "i love this movie!"

    keras_inputs = tokenize_sentence(tokenizer, text)
    keras_inputs = {k: tf.expand_dims(v, axis=0) for k, v in keras_inputs.items()}
    model = tf.keras.models.load_model('../llama_sentiment_analysis/model/llama_32_1b_inst.h5')
    # --------------------------------- Evaluating Inference -------------------------------------
    outputs = model(keras_inputs)

    # last_token_logits = outputs[0, -1]
    # pred_token_id = np.argmax(last_token_logits)
    # pred_token = tokenizer.decode([pred_token_id]).strip().lower()

    pred_token = tokenizer.decode(np.argmax(outputs, axis=2)[0], skip_special_tokens=True).split()[-1]

    print(f'sentence: {text}, prediction: {pred_token}')


def simple_inference():
    # ----------------------------------------- Input Preparation --------------------------
    tokenizer = AutoTokenizer.from_pretrained("../llama_sentiment_analysis/llama_32_1b")
    tokenizer.pad_token = tokenizer.eos_token
    text = "i love this movie!"

    keras_inputs = tokenize_sentence_reg(tokenizer, text)
    keras_inputs = {k: tf.expand_dims(v, axis=0) for k, v in keras_inputs.items()}
    model = tf.keras.models.load_model('../llama_sentiment_analysis/llama_32_1b/llama_32_1b.h5')
    # --------------------------------- Evaluating Inference -------------------------------------
    outputs = model(keras_inputs)

    # last_token_logits = outputs[0, -1]
    # pred_token_id = np.argmax(last_token_logits)
    # pred_token = tokenizer.decode([pred_token_id]).strip().lower()

    pred_token = tokenizer.decode(np.argmax(outputs, axis=2)[0], skip_special_tokens=True).split()[-1]

    print(f'sentence: {text}, prediction: {pred_token}')


if __name__ == '__main__':
    simple_inference_instruct()