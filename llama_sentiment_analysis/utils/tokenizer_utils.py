import numpy as np
import tensorflow as tf

def tokenize_sentence(tokenizer, text):
    prompt = tokenizer.apply_chat_template(
        [{"role": "user",
          "content": f"What is the sentiment of this sentence: \"{text}\"? Respond with 'Positive', 'Negative' or 'Neutral' only."}],
        add_generation_prompt=True,
        return_tensors="np"
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
