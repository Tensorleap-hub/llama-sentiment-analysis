from transformers import AutoTokenizer
import numpy as np
import tensorflow as tf

def simple_inference():
    # ----------------------------------------- Input Preparation --------------------------
    tokenizer = AutoTokenizer.from_pretrained("../model")
    tokenizer.pad_token = tokenizer.eos_token
    text = "i love this movie!"
    prompt = tokenizer.apply_chat_template(
        [{"role": "user",
          "content": f"What is the sentiment of this sentence: \"{text}\"? Respond with 'positive' or 'negative' only."}],
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
    keras_inputs = {k: tf.convert_to_tensor(v) for k, v in model_inputs.items()}

    model = tf.keras.models.load_model('../model/temp.h5')
    # --------------------------------- Evaluating Inference -------------------------------------
    outputs = model(keras_inputs)
    last_token_logits = outputs[0, -1]
    pred_token_id = np.argmax(last_token_logits)
    pred_token = tokenizer.decode([pred_token_id]).strip().lower()
    print(f'sentence: {text}, prediction: {pred_token}')

if __name__ == '__main__':
    simple_inference()