import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from sklearn.model_selection import train_test_split
from pdb import set_trace
import sys
from torch.nn.utils.rnn import pad_sequence
from datatypes import *
from model import *


def read_data(configs: dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    '''
    In the test_case_verdict_x_y field 0 means correct, 1 means wa, 2 means RTE, 3 means TLE
    '''
    # load dataset
    dataset = pd.read_pickle(configs.data_path + '/dataset.pkl')

    allowed_problemIDs = ['20']
    dataset = dataset[dataset['problemID'].isin(allowed_problemIDs)]

    #split the dataset into two, one for is_similar = True and the other for is_similar = False
    dataset_true = dataset[dataset['is_similar'] == True]
    dataset_false = dataset[dataset['is_similar'] == False]

    #sample the dataset_false to have the same number of rows as dataset_true
    # dataset_false = dataset_false.sample(n=configs.true_false_ratio * dataset_true.shape[0])

    #sample the dataset_false to select good negative samples
    dataset_false = sample_good_negatives(dataset_true, dataset_false, n = configs.true_false_ratio * dataset_true.shape[0])

    #prepare the dataset
    if configs.loss_fn in ['ContrastiveLoss', 'CosineSimilarityLoss', 'MultipleNegativesRankingLoss']:
        dataset = pd.concat([dataset_true, dataset_false])
    else:
        print('No dataset for this loss function')
        return None
    # elif configs.loss_fn == 'TripletLoss':

        
    ## if only testing, subsample part of dataset
    if configs.testing:
        dataset = dataset.sample(n=configs.testing_size)
        
    trainset, testset = train_test_split(dataset, test_size=configs.test_size, random_state=configs.seed)
    validset, testset = train_test_split(testset, test_size=0.5, random_state=configs.seed)

    return trainset, validset, testset


def sample_good_negatives(dataset_true: pd.DataFrame, dataset_false: pd.DataFrame, n: int) -> pd.DataFrame:
    set_AB = set(dataset_true[['problemID', 'test_case_verdict_i_1']].itertuples(index=False, name=None))
    set_AC = set(dataset_true[['problemID', 'test_case_verdict_j_1']].itertuples(index=False, name=None))

    filtered_dataset_false_AB = dataset_false[dataset_false[['problemID', 'test_case_verdict_i_1']].apply(tuple, axis=1).isin(set_AB)]
    filtered_dataset_false_AC = dataset_false[dataset_false[['problemID', 'test_case_verdict_j_1']].apply(tuple, axis=1).isin(set_AC)]

    merged_dataset = pd.concat([filtered_dataset_false_AB, filtered_dataset_false_AC]).drop_duplicates()

    # Check if we have enough negatives and sample accordingly
    if len(merged_dataset) < n:
        print(f"Warning: Requested {n} negatives but only found {len(merged_dataset)}.")
        return None
    else:
        return merged_dataset.sample(n=n)

def make_dataloader(dataset: pd.DataFrame, collate_fn: callable, configs: dict, n_workers: int = 0, train: bool = True) -> torch.utils.data.DataLoader:
    shuffle = train and not configs.testing
    pytorch_dataset = CERDataset(dataset)
    return torch.utils.data.DataLoader(pytorch_dataset, collate_fn=collate_fn, shuffle=shuffle, batch_size=configs.batch_size, num_workers=n_workers)

class CERDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe: pd.DataFrame):
        self.data = dataframe

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        return {
            'A1': row['code_i_1'],
            'A2': row['code_i_2'],
            'B1': row['code_j_1'],
            'B2': row['code_j_2'],
            'label': 1 if row['is_similar'] else 0,
        }
    
class CollateForCER(object):
    def __init__(self, tokenizer: tokenizer, configs: dict, device: torch.device):
        self.tokenizer = tokenizer
        self.configs = configs
        self.device = device

    def __call__(self, batch: List[Dict[str, Union[str, int]]]) -> Dict[str, torch.Tensor]:
        # Create a single list where each A1, A2, B1, and B2 will be concatenated consecutively
        concatenated_inputs = []
        labels = []

        # for item in batch:
        #     concatenated_inputs.append(item['A1'])  # Add A1
        #     concatenated_inputs.append(item['A2'])  # Add A2
        #     concatenated_inputs.append(item['B1'])  # Add B1
        #     concatenated_inputs.append(item['B2'])  # Add B2
        #     labels.append(item['label'])
        A1 = [item['A1'] for item in batch]
        A2 = [item['A2'] for item in batch]
        B1 = [item['B1'] for item in batch]
        B2 = [item['B2'] for item in batch]
        labels = [item['label'] for item in batch]
        concatenated_inputs = A1 + A2 + B1 + B2

        # Need to tokenize here for efficiency
        # inputs = self.tokenizer(code, return_tensors="pt", padding=True, truncation=True).to(self.device)
        # concatenated_inputs = tokenizer(concatenated_inputs, return_tensors='pt', padding=True, truncation=True)
        
        return {
            'inputs': concatenated_inputs,  # This is a single list containing A1, A2, B1, B2 in order
            'labels': torch.tensor(labels)
        }
   

# # def make_pytorch_dataset(dataset_split, dataset_full, do_lstm_dataset=True):
# def make_pytorch_dataset(dataset: pd.DataFrame) -> List[Dict[str, Union[str, int]]]:
#     '''
#     convert the pandas dataframe into dataset format that pytorch dataloader takes
#     the resulting format is a list of dictionaries
#     '''
#     #loop through dataset dataframe and append each row as a dictionary to a list
#     cer_dataset = []
#     for index, row in dataset.iterrows():
#         # print(row)
#         cer_dataset.append({
#             'A1': row['code_i_1'],
#             'A2': row['code_i_2'],
#             'B1': row['code_j_1'],
#             'B2': row['code_j_2'],
#             'label': 1 if row['is_similar'] == True else 0,
#         }) 
#     return cer_dataset



# # def make_dataloader(dataset_split, dataset_full, collate_fn, configs, n_workers=0, do_lstm_dataset=True, train=True):
# def make_dataloader(dataset: pd.DataFrame, collate_fn: Callable, configs: dict, n_workers: int = 0, train: bool = True) -> torch.utils.data.DataLoader:
#     shuffle = True if train else False
#     if configs.testing:
#         shuffle = False
#     dataset = make_pytorch_dataset(dataset=dataset)
#     # data_loader = torch.utils.data.DataLoader(dataset, collate_fn=collate_fn, shuffle=shuffle, batch_size=configs.batch_size, num_workers=n_workers)
#     data_loader = torch.utils.data.DataLoader(dataset, shuffle=shuffle, batch_size=configs.batch_size, num_workers=n_workers)
#     return data_loader
    
# class CollateForCER(object):
#     def __init__(self, tokenizer: tokenizer, configs: dict, device: torch.device):
#         self.tokenizer = tokenizer
#         self.configs = configs
#         self.device = device

#     def __call__(self, batch: List[Dict[str, Union[str, int]]]) -> Dict[str, torch.Tensor]:
#         input_ids_list = []
#         attention_mask_list = []
        
#         for b in batch:
#             # Tokenize each input sequence and concatenate them
#             inputs = self.tokenizer(
#                 [b['A1'], b['A2'], b['B1'], b['B2']],
#                 return_tensors='pt',
#                 padding=True,
#                 truncation=True
#             )
#             input_ids = torch.cat([inputs['input_ids'][i] for i in range(len(inputs['input_ids']))], dim=0)
#             attention_mask = torch.cat([inputs['attention_mask'][i] for i in range(len(inputs['attention_mask']))], dim=0)
            
#             input_ids_list.append(input_ids)
#             attention_mask_list.append(attention_mask)

#         # Find the maximum sequence length in the batch
#         max_len = max([seq.size(0) for seq in input_ids_list])
        
#         # Pad sequences to the maximum length
#         input_ids_padded = torch.stack([torch.nn.functional.pad(seq, (0, max_len - seq.size(0)), value=self.tokenizer.pad_token_id) for seq in input_ids_list])
#         attention_mask_padded = torch.stack([torch.nn.functional.pad(seq, (0, max_len - seq.size(0)), value=0) for seq in attention_mask_list])

#         # Move tensors to the specified device
#         input_ids = input_ids_padded.to(self.device)
#         attention_mask = attention_mask_padded.to(self.device)
        
#         # Stack the labels and move to the specified device
#         labels = torch.stack([torch.tensor(b['label'], dtype=torch.long) for b in batch]).to(self.device)
        
#         return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}

