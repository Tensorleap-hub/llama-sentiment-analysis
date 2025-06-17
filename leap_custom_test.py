import os

import psutil
from tqdm import tqdm

from leap_binder import *
from llama_sentiment_analysis.llama import CE_loss


def print_memory_usage():
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / 1024 ** 2  # in MB
    print(f"Current memory usage: {mem_mb:.2f} MB")

def check_custom_integration():
    LOAD_MODEL = True
    PLOT = True
    check_generic = True

    if check_generic:
        leap_binder.check()

    print("Starting custom tests")

    # Load Data
    train, val, test = preprocess_func()

    if LOAD_MODEL:
        H5_MODEL_PATH = "llama_sentiment_analysis/model/llama_32_1b_inst.h5"
        dir_path = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(dir_path, H5_MODEL_PATH)

        model = tf.keras.models.load_model(model_path)
    else:
        raise Exception("Cant load from HF need to convert to h5 first")

    # for sub in tqdm([train, val, test, ul]):
    for sub in tqdm([train]):
        for i in tqdm(range(1, sub.length), desc="Samples"):
            gt = gt_encoder(i, sub)#[None, ...]
            inputs = {}
            inputs["input_ids"] = input_ids(i, sub)[None, ...]
            inputs["position_ids"] = position_ids(i, sub)[None, ...]
            inputs["attention_mask"] = input_attention_mask(i, sub)[None, ...]

            attention_masks_raw = input_attention_mask_wrapped(i, sub)

            pred = model(inputs)
            pred_token_id, pred_token_text, last_token_index = get_label_from_prediction(inputs["attention_mask"], pred)

            scores = calc_metrics(gt, attention_masks_raw, pred.numpy())
            loss = CE_loss(gt, attention_masks_raw, pred.numpy())
            print(loss, scores, tokenizer.decode(np.array(inputs["input_ids"][0], dtype=np.int), skip_special_tokens=True), tokenizer.convert_ids_to_tokens(int(gt[0])), pred_token_text)
            print_memory_usage()

    print("Done")

if __name__ == '__main__':
    check_custom_integration()