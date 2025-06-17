import os
from typing import Tuple, Optional, List, Union

import numpy
from datasets import load_dataset, load_from_disk
import numpy as np
from datasets import Dataset

from llama_sentiment_analysis.config import CONFIG

SEED=42
np.random.seed(SEED)

def load_data() -> Tuple[Dataset, Dataset, Dataset]:
    dir_path = os.path.dirname(os.path.abspath(__file__))
    ds_train = load_from_disk(os.path.join(dir_path, "data/glue_sst_train")).shuffle(seed=SEED)
    ds_val = load_from_disk(os.path.join(dir_path, "data/glue_sst_val")).shuffle(seed=SEED)
    ds_test = load_from_disk(os.path.join(dir_path, "data/glue_sst_test")).shuffle(seed=SEED)
    return ds_train, ds_val, ds_test

def get_dataset_label_map():
    return {0: "Negative", 1: "Positive", -1: "Neutral"}

def downsample_hf_dataset(dataset: Dataset, size: int, indices: Optional[Union[List[int], np.ndarray]] = None) -> Dataset:
    indices = np.array(indices) if indices is not None else np.arange(len(dataset))
    # Generate random indices without replacement
    random_indices = np.random.choice(indices, size=min(size, len(dataset)), replace=False)
    # Subsample the dataset using the selected indices
    subset_dataset = dataset.select(random_indices)
    return subset_dataset



