'''
Create AUDIT-C Scoring dataset for a quick and dirty pilot run using
`AUD_LLM_CX_04052024.csv`.

206,864 samples are for training, 500 for val, and 5,000 for test.
'''

from datasets import load_dataset, DatasetDict
from utils import CSV_PATH, DATASET_PATH, SEED


dataset = load_dataset('csv', data_files=CSV_PATH)
dataset = dataset.shuffle(seed=SEED)

train_test_split = dataset['train'].train_test_split(test_size=5500, seed=SEED)
val_test_split = train_test_split['test'].train_test_split(test_size=5000, seed=SEED)

split_datasets = DatasetDict({
    'train': train_test_split['train'],  # 206,864
    'val': val_test_split['train'],  # 500
    'test': val_test_split['test']  # 5,000
})

split_datasets.save_to_disk(DATASET_PATH)
