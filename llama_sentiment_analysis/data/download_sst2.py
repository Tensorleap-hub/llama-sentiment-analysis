# Getting data from HF, do it once, save the data and use it locally
from datasets import load_dataset

ds_train = load_dataset("glue", "sst2", split="train")
ds_val = load_dataset("glue", "sst2",  split="validation")
ds_test = load_dataset("glue", "sst2",  split="test")
ds_train.save_to_disk("./glue_sst_train")
ds_val.save_to_disk("./glue_sst_val")
ds_test.save_to_disk("./glue_sst_test")