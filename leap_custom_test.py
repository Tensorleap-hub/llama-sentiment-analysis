import os
from tqdm import tqdm

from leap_binder import *

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
        H5_MODEL_PATH = "model/llama_32_1b_inst.h5"
        dir_path = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(dir_path, H5_MODEL_PATH)

        model = tf.keras.models.load_model(model_path)
    else:
        raise Exception("Cant load from HF need to convert to h5 first")

    # for sub in tqdm([train, val, test, ul]):
    for sub in tqdm([val]):
        for i in tqdm(range(1, sub.length), desc="Samples"):
            gt = gt_encoder(i, sub)#[None, ...]
            inputs = {}
            inputs["input_ids"] = input_ids(i, sub)[None, ...]
            inputs["position_ids"] = position_ids(i, sub)[None, ...]
            inputs["attention_mask"] = input_attention_mask(i, sub)[None, ...]

            pred = model(inputs)
            pred = pred[:, -1, :]

            batched_gt = gt[None, ...]

            scores = calc_metrics(np.array(batched_gt), pred.numpy())
            loss = CE_loss(gt, pred[0])
            print(loss, scores, tokenizer.decode(np.array(inputs["input_ids"][0], dtype=np.int), skip_special_tokens=True), tokenizer.convert_ids_to_tokens(int(gt[0])), tokenizer.convert_ids_to_tokens(int(np.argmax(pred[0]))))

    print("Done")

if __name__ == '__main__':
    check_custom_integration()