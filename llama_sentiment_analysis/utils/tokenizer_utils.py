import numpy as np
import tensorflow as tf


def tokenize_sentence_new(tokenizer, text, max_length=512):
    # 1. Format prompt using chat template
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user",
          "content": f"What is the sentiment of this sentence: \"{text}\"? Respond with one word -  'Positive', 'Negative' or 'Neutral' only."}],
        add_generation_prompt=True,
        tokenize=False  # Get plain string to tokenize manually
    )

    # 2. Tokenize with truncation/padding to fixed max length
    encoded = tokenizer(
        prompt_text,
        return_tensors="np",
        padding="max_length",
        truncation=True,
        max_length=max_length
    )

    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    position_ids = np.arange(max_length)[None, :]  # Always same size

    model_inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids
    }

    keras_inputs = {k: tf.convert_to_tensor(v)[0] for k, v in model_inputs.items()}
    return keras_inputs


def tokenize_sentence(tokenizer, text):
    prompt = tokenizer.apply_chat_template(
        [{"role": "user",
          "content": f"What is the sentiment of this sentence: \"{text}\"? Respond with 'Positive', 'Negative' or 'Neutral' only."}],
        add_generation_prompt=True,
        return_tensors="np",
        max_length=512,
        padding="max_length",
        truncation=True,
    )
    input_ids = prompt
    attention_mask = (input_ids != tokenizer.pad_token_id).astype(np.int64)
    position_ids = np.arange(input_ids.shape[1])[None, :]
    model_inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids
    }
    keras_inputs = {k: tf.convert_to_tensor(v)[0] for k, v in model_inputs.items()}
    return keras_inputs


def tokenize_sentence_reg(tokenizer, text):
    # Construct a direct prompt for regular LLMs
    prompt_text = f"What is the sentiment of this sentence: \"{text}\"? Respond with 'positive' or 'negative' only. The sentiment is"

    # Tokenize normally (no chat template)
    encoding = tokenizer(prompt_text, return_tensors="pt", padding='max_length', truncation=True, max_length=128)

    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]
    position_ids = np.arange(input_ids.shape[1])[None, :]

    model_inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids
    }
    keras_inputs = {k: tf.convert_to_tensor(v)[0] for k, v in model_inputs.items()}
    return keras_inputs
