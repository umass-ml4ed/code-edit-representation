import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from sklearn.model_selection import train_test_split
from pdb import set_trace
import sys
from torch.nn.utils.rnn import pad_sequence
from datatypes import *


def read_data(configs: dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    '''
    In the test_case_verdict_x_y field 0 means correct, 1 means wa, 2 means RTE, 3 means TLE
    '''
    # load dataset
    dataset = pd.read_pickle(configs.data_path + '/dataset.pkl')

    #split the dataset into two, one for is_similar = True and the other for is_similar = False
    dataset_true = dataset[dataset['is_similar'] == True]
    dataset_false = dataset[dataset['is_similar'] == False]

    #sample the dataset_false to have the same number of rows as dataset_true
    # dataset_false = dataset_false.sample(n=configs.true_false_ratio * dataset_true.shape[0])

    #sample the dataset_false to select good negative samples
    dataset_false = sample_good_negatives(dataset_true, dataset_false, n = configs.true_false_ratio * dataset_true.shape[0])

    #concatenate the two datasets
    dataset = pd.concat([dataset_true, dataset_false])
        
    ## if only testing, subsample part of dataset
    if configs.testing:
        dataset = dataset.sample(n=16)
        
    trainset, testset = train_test_split(dataset, test_size=configs.test_size, random_state=configs.seed)
    validset, testset = train_test_split(testset, test_size=0.5, random_state=configs.seed)

    return trainset, validset, testset, dataset#, students


def sample_good_negatives(dataset_true: pd.DataFrame, dataset_false: pd.DataFrame, n: int) -> pd.DataFrame:
    # Convert the relevant columns of dataset_true to sets of tuples
    set_AB = set(dataset_true[['problemID', 'test_case_verdict_i_1']].itertuples(index=False, name=None))
    set_AC = set(dataset_true[['problemID', 'test_case_verdict_j_1']].itertuples(index=False, name=None))

    # Filter dataset_false using sets for problemID and test_case_verdict_i_1
    filtered_dataset_false_AB = dataset_false[dataset_false[['problemID', 'test_case_verdict_i_1']].apply(tuple, axis=1).isin(set_AB)]

    # Filter dataset_false using sets for problemID and test_case_verdict_j_1
    filtered_dataset_false_AC = dataset_false[dataset_false[['problemID', 'test_case_verdict_j_1']].apply(tuple, axis=1).isin(set_AC)]

    # Combine the two filtered datasets
    merged_dataset = pd.concat([filtered_dataset_false_AB, filtered_dataset_false_AC])

    # Remove duplicates
    unique_merged_dataset = merged_dataset.drop_duplicates()

    #sample to match the ratio of dataset_true
    dataset_false = unique_merged_dataset.sample(n=n)

    return dataset_false

# def make_pytorch_dataset(dataset_split, dataset_full, do_lstm_dataset=True):
def make_pytorch_dataset(dataset: pd.DataFrame) -> List[Dict[str, Union[str, int]]]:
    '''
    convert the pandas dataframe into dataset format that pytorch dataloader takes
    the resulting format is a list of dictionaries
    '''
    #loop through dataset dataframe and append each row as a dictionary to a list
    cer_dataset = []
    for index, row in dataset.iterrows():
        # print(row)
        cer_dataset.append({
            'A1': row['code_i_1'],
            'A2': row['code_i_2'],
            'B1': row['code_j_1'],
            'B2': row['code_j_2'],
            'label': 1 if row['is_similar'] == True else 0,
        }) 
    return cer_dataset



# def make_dataloader(dataset_split, dataset_full, collate_fn, configs, n_workers=0, do_lstm_dataset=True, train=True):
def make_dataloader(dataset: pd.DataFrame, collate_fn: Callable, configs: dict, n_workers: int = 0, train: bool = True) -> torch.utils.data.DataLoader:
    shuffle = True if train else False
    
    dataset = make_pytorch_dataset(dataset=dataset)
    # data_loader = torch.utils.data.DataLoader(dataset, collate_fn=collate_fn, shuffle=shuffle, batch_size=configs.batch_size, num_workers=n_workers)
    data_loader = torch.utils.data.DataLoader(dataset, shuffle=shuffle, batch_size=configs.batch_size, num_workers=n_workers)
    return data_loader
    


class CollateForCER(object):
    def __init__(self, tokenizer: tokenizer, configs: dict, device: torch.device):
        self.tokenizer = tokenizer
        self.configs = configs
        self.device = device

    def __call__(self, batch: List[Dict[str, Union[str, int]]]) -> Dict[str, torch.Tensor]:
        input_ids_list = []
        attention_mask_list = []
        
        for b in batch:
            # Tokenize each input sequence and concatenate them
            inputs = self.tokenizer(
                [b['A1'], b['A2'], b['B1'], b['B2']],
                return_tensors='pt',
                padding=True,
                truncation=True
            )
            input_ids = torch.cat([inputs['input_ids'][i] for i in range(len(inputs['input_ids']))], dim=0)
            attention_mask = torch.cat([inputs['attention_mask'][i] for i in range(len(inputs['attention_mask']))], dim=0)
            
            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)

        # Find the maximum sequence length in the batch
        max_len = max([seq.size(0) for seq in input_ids_list])
        
        # Pad sequences to the maximum length
        input_ids_padded = torch.stack([torch.nn.functional.pad(seq, (0, max_len - seq.size(0)), value=self.tokenizer.pad_token_id) for seq in input_ids_list])
        attention_mask_padded = torch.stack([torch.nn.functional.pad(seq, (0, max_len - seq.size(0)), value=0) for seq in attention_mask_list])

        # Move tensors to the specified device
        input_ids = input_ids_padded.to(self.device)
        attention_mask = attention_mask_padded.to(self.device)
        
        # Stack the labels and move to the specified device
        labels = torch.stack([torch.tensor(b['label'], dtype=torch.long) for b in batch]).to(self.device)
        
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}
