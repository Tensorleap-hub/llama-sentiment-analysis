from typing import Dict, Any, List

from code_loader import leap_binder
from code_loader.contract.enums import LeapDataType
from code_loader.contract.datasetclasses import PreprocessResponse
from code_loader.contract.visualizer_classes import LeapText
from code_loader.inner_leap_binder.leapbinder_decorators import (tensorleap_preprocess, tensorleap_unlabeled_preprocess,
                                                                 tensorleap_input_encoder, tensorleap_gt_encoder,
                                                                 tensorleap_metadata, tensorleap_custom_visualizer)

from llama_sentiment_analysis.dataset import load_data, downsample_hf_dataset, get_dataset_label_map
from llama_sentiment_analysis.llama import tokenize_and_align_labels, get_labels_ids_map, tokenizer
from llama_sentiment_analysis.utils.metrics import *

@tensorleap_preprocess()
def preprocess_func() -> List[PreprocessResponse]:

    ds_train, ds_val, ds_test = load_data()

    ds_train = downsample_hf_dataset(ds_train, CONFIG["train_size"])
    ds_val = downsample_hf_dataset(ds_val, CONFIG["val_size"])

    idx = len(ds_test)//2
    ds_test = ds_test.select(np.arange(idx))
    ds_test = downsample_hf_dataset(ds_test, CONFIG["test_size"])

    train = PreprocessResponse(length=len(ds_train), data={'ds': ds_train, 'subset': 'train'})
    val = PreprocessResponse(length=len(ds_val), data={'ds': ds_val, 'subset': 'val'})
    test = PreprocessResponse(length=len(ds_test), data={'ds': ds_test, 'subset': 'test'})
    response = [train, val, test]
    return response


@tensorleap_unlabeled_preprocess()
def preprocess_func_ul() -> List[PreprocessResponse]:
    _, _, ds_test = load_data()
    idx = len(ds_test)//2
    ds_test = ds_test.select(np.arange(idx, len(ds_test)))
    ds_test = downsample_hf_dataset(ds_test, CONFIG["ul_size"])
    response = PreprocessResponse(length=len(ds_test), data={'ds': ds_test, 'subset': 'ul'})
    return response


def input_encoder(idx: int, preprocess: PreprocessResponse) -> np.ndarray:
    sample = preprocess.data['ds'][idx:idx+1] #['tokens']
    tokenized_inputs = tokenize_and_align_labels(sample)
    return tokenized_inputs


@tensorleap_input_encoder(name="input_ids")
def input_ids(idx: int, preprocess: PreprocessResponse) -> np.ndarray:
    inputs = input_encoder(idx, preprocess)
    inputs = inputs["input_ids"]
    return inputs.numpy().astype(np.float32)


@tensorleap_input_encoder(name="position_ids")
def position_ids(idx: int, preprocess: PreprocessResponse) -> np.ndarray:
    inputs = input_encoder(idx, preprocess)
    inputs = inputs["position_ids"]
    return inputs.numpy().astype(np.float32)


@tensorleap_input_encoder(name="attention_mask")
def input_attention_mask(idx: int, preprocess: PreprocessResponse) -> np.ndarray:
    inputs = input_encoder(idx, preprocess)
    inputs = inputs["attention_mask"]
    return inputs.numpy().astype(np.float32)


@tensorleap_gt_encoder(name="attention_mask_wrapped")
def input_attention_mask_wrapped(idx: int, preprocess: PreprocessResponse) -> np.ndarray:
    inputs = input_encoder(idx, preprocess)
    inputs = inputs["attention_mask"]
    return inputs.numpy().astype(np.float32)


@tensorleap_gt_encoder(name="label_token_id")
def gt_encoder(idx: int, preprocess: PreprocessResponse) -> np.ndarray:
    tokenized_inputs = input_encoder(idx, preprocess)   # get tokenized labels
    labels = tokenized_inputs["labels"]
    label_map = get_dataset_label_map()
    label_ids_map = get_labels_ids_map()
    text_labels = [label_map[label] for label in labels]
    token_id_label = [label_ids_map[text_label] for text_label in text_labels]
    return np.array(token_id_label).astype(np.float32)


@tensorleap_metadata('metadata_sample_label')
def metadata_sample_index(idx: int, preprocess: PreprocessResponse) -> int:
    tokenized_inputs = input_encoder(idx, preprocess)
    label_token_id = tokenized_inputs["labels"]
    return int(label_token_id[0])

@tensorleap_custom_visualizer('sentence_visualizer', LeapDataType.Text)
def sentence_visualizer(input_ids_arr) -> LeapText:
    return LeapText(tokenizer.decode(np.array(input_ids_arr[0], dtype=np.int32), skip_special_tokens=True).split(' '))

@tensorleap_custom_visualizer('gt_label_visualizer', LeapDataType.Text)
def gt_label_visualizer(label_token_id: np.float32) -> LeapText:
    return LeapText([tokenizer.convert_ids_to_tokens(int(label_token_id))])

@tensorleap_custom_visualizer('pred_label_visualizer', LeapDataType.Text)
def pred_label_visualizer(attention_masks, prediction) -> LeapText:
    pred_token_id, pred_token_text, last_token_index = get_label_from_prediction(attention_masks, prediction)
    return LeapText([pred_token_text])

if __name__ == "__main__":
    leap_binder.check()
