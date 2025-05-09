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
from torch.utils.data import DataLoader, Dataset

from sklearn.model_selection import train_test_split
import pandas as pd
from typing import Tuple

def read_data_split_by_student_ID(configs: dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Reads and splits the dataset ensuring no students are reused across train, test, and valid sets.

    Parameters:
    configs (dict): Configuration dictionary with dataset path, allowed problem list, and other options.

    Returns:
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Train, validation, and test datasets.
    """
    print('Splitting Train/Test/Valid based on Student IDs')
    # Load dataset
    dataset = pd.read_pickle(configs['data_path'])

    # Filter allowed problem IDs
    allowed_problemIDs = configs['allowed_problem_list']
    dataset = dataset[dataset['problemID'].isin(allowed_problemIDs)]

    #split the dataset into two, one for is_similar = True and the other for is_similar = False
    dataset_true = dataset[dataset['is_similar'] == True]
    dataset_false = dataset[dataset['is_similar'] == False]

    #sample the dataset_false to have the same number of rows as dataset_true
    dataset_false = dataset_false.sample(n=configs.true_false_ratio * dataset_true.shape[0])

    #prepare the dataset
    if configs.loss_fn in ['ContrastiveLoss', 'CosineSimilarityLoss', 'MultipleNegativesRankingLoss']:
        dataset = pd.concat([dataset_true, dataset_false])
    else:
        print('No dataset for this loss function')
        return None

    # Extract unique students
    all_students = pd.unique(
        dataset[['studentID_1', 'studentID_2']].values.ravel()
    )
    for test_size in np.arange(configs['test_size'], .6, .1):
    # Split the students into train, valid, and test sets
        train_students, test_valid_students = train_test_split(all_students, test_size=test_size, random_state=configs['seed'])
        valid_students, test_students = train_test_split(test_valid_students, test_size=0.5, random_state=configs['seed'])

        # Filter the dataset to ensure no overlap
        def filter_dataset(dataset, allowed_students):
            return dataset[
                dataset['studentID_1'].isin(allowed_students) &
                dataset['studentID_2'].isin(allowed_students)
            ]

        trainset = filter_dataset(dataset, train_students)
        validset = filter_dataset(dataset, valid_students)
        testset = filter_dataset(dataset, test_students)

        print(len(trainset), len(testset), len(validset))

        if len(testset) >= len(trainset) * configs['test_size'] * 0.2 and len(validset) >= len(trainset) * configs['test_size'] * 0.2: break

    # Save the current dataset (optional)
    dataset.to_pickle('data/current_dataset.pkl')

    return trainset, validset, testset


def read_data_problem_exclusive(configs: dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Reads and splits the dataset ensuring no code pairs overlap between train and test/valid sets.

    Parameters:
    configs (dict): Configuration dictionary with dataset path, allowed problem list, and other options.

    Returns:
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Train, validation, and test datasets.
    """
    print('Splitting train/test/valid set by unique code IDs')
    # Load dataset
    dataset = pd.read_pickle(configs.data_path)

    # Filter allowed problem IDs
    allowed_problemIDs = configs['allowed_problem_list']
    dataset = dataset[dataset['problemID'].isin(allowed_problemIDs)]

    # Create unique identifiers for each code pair
    dataset['pair_id'] = (
        dataset['codeID_i_1'].astype(str) + "_" + 
        dataset['codeID_j_1'].astype(str) + "_" +
        dataset['codeID_i_2'].astype(str) + "_" +
        dataset['codeID_j_2'].astype(str)
    )

    # Split the dataset into true and false cases
    dataset_true = dataset[dataset['is_similar'] == True]
    dataset_false = dataset[dataset['is_similar'] == False]

    # Balance the dataset
    dataset_false = dataset_false.sample(n=configs['true_false_ratio'] * len(dataset_true), random_state=configs['seed'])
    dataset_balanced = pd.concat([dataset_true, dataset_false])

    # Split based on unique pairs
    unique_pairs = dataset_balanced['pair_id'].unique()
    train_pairs, test_valid_pairs = train_test_split(unique_pairs, test_size=configs['test_size'], random_state=configs['seed'])
    valid_pairs, test_pairs = train_test_split(test_valid_pairs, test_size=0.5, random_state=configs['seed'])

    # Filter datasets based on the splits
    trainset = dataset_balanced[dataset_balanced['pair_id'].isin(train_pairs)]
    validset = dataset_balanced[dataset_balanced['pair_id'].isin(valid_pairs)]
    testset = dataset_balanced[dataset_balanced['pair_id'].isin(test_pairs)]

    # Drop pair_id to avoid leaking identifiers
    trainset = trainset.drop(columns=['pair_id'])
    validset = validset.drop(columns=['pair_id'])
    testset = testset.drop(columns=['pair_id'])

    # Save the current dataset (optional)
    dataset_balanced.to_pickle('data/current_dataset.pkl')

    return trainset, validset, testset

def read_data(configs: dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    '''
    In the test_case_verdict_x_y field 0 means correct, 1 means wa, 2 means RTE, 3 means TLE
    '''
    print('Splitting randomly')
    # load dataset
    dataset = pd.read_pickle(configs.data_path)

    allowed_problemIDs = configs.allowed_problem_list
    dataset = dataset[dataset['problemID'].isin(allowed_problemIDs)]

    #split the dataset into two, one for is_similar = True and the other for is_similar = False
    dataset_true = dataset[dataset['is_similar'] == True]
    dataset_false = dataset[dataset['is_similar'] == False]

    #sample the dataset_false to have the same number of rows as dataset_true
    dataset_false = dataset_false.sample(n=configs.true_false_ratio * dataset_true.shape[0])

    #sample the dataset_false to select good negative samples
    # if configs.testing == False:
    #     dataset_false = sample_good_negatives(dataset_true, dataset_false, n = configs.true_false_ratio * dataset_true.shape[0])

    #prepare the dataset
    if configs.loss_fn in ['ContrastiveLoss', 'CosineSimilarityLoss', 'MultipleNegativesRankingLoss']:
        dataset = pd.concat([dataset_true, dataset_false])
    else:
        print('No dataset for this loss function')
        return None
    # elif configs.loss_fn == 'TripletLoss':

        
    ## if only testing, subsample part of dataset
    # if configs.testing:
    #     dataset = dataset.sample(n=configs.testing_size)
    
    trainset = dataset 
    dataset.to_pickle('data/current_dataset.pkl')
    testset = validset = None
    # if configs.testing == False:
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

def make_dataloader_experiment(dataset: pd.DataFrame, collate_fn: callable, configs: dict, n_workers: int = 0, train: bool = True) -> torch.utils.data.DataLoader:
    shuffle = False
    pytorch_dataset = CERDataset(dataset)
    return torch.utils.data.DataLoader(pytorch_dataset, collate_fn=collate_fn, shuffle=shuffle, batch_size=configs.batch_size_exp, num_workers=n_workers)

class CERDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe: pd.DataFrame):
        self.data = dataframe

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        return {
            'A1': row['code_i_1'],
            'A2': row['code_j_1'],
            'B1': row['code_i_2'],
            'B2': row['code_j_2'],
            'label': 1 if row['is_similar'] else 0,
            'A1_mask': row['test_case_verdict_i_1'],
            'A2_mask': row['test_case_verdict_j_1'],
            'B1_mask': row['test_case_verdict_i_2'],
            'B2_mask': row['test_case_verdict_j_2'],
            'problemID': row['problemID'],
            'studentID1': row['studentID_1'],
            'studentID2': row['studentID_2'],
        }
    
class CollateForCER(object):
    def __init__(self, tokenizer: tokenizer, configs: dict, device: torch.device):
        self.tokenizer = tokenizer
        self.configs = configs
        self.device = device

    def __call__(self, batch: List[Dict[str, Union[str, int]]]) -> Dict[str, torch.Tensor]:
        # Create a single list where each A1, A2, B1, and B2 will be concatenated consecutively
        # concatenated_inputs = []
        # labels = []
        A1 = [item['A1'] for item in batch]
        A2 = [item['A2'] for item in batch]
        B1 = [item['B1'] for item in batch]
        B2 = [item['B2'] for item in batch]
        labels = [item['label'] for item in batch]
        concatenated_inputs = A1 + A2 + B1 + B2

        A1_mask = [item['A1_mask'] for item in batch]
        A2_mask = [item['A2_mask'] for item in batch]
        B1_mask = [item['B1_mask'] for item in batch]
        B2_mask = [item['B2_mask'] for item in batch]
        concatenated_inputs_mask = A1_mask + A2_mask + B1_mask + B2_mask
        
        problemIDs = [int(item['problemID']) for item in batch]
        studentIDs = [(item['studentID1'], item['studentID2']) for item in batch]
        return {
            'inputs': concatenated_inputs,  # This is a single list containing A1, A2, B1, B2 in order
            'labels': torch.tensor(labels),
            'masks': concatenated_inputs_mask,
            'problemIDs': problemIDs,
            'studentIDs': studentIDs,
        }

class DecoderFineTuneDataset(Dataset):
    def __init__(self, dataframe, encoder_model, tokenizer, device):
        self.data = dataframe
        self.encoder_model = encoder_model
        self.tokenizer = tokenizer
        self.device = device

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        
        # Get the four code instances
        code_instances = [row['code_i_1'], row['code_j_1'], row['code_i_2'], row['code_j_2']]
        
        # Tokenize each code instance for decoding targets
        target_ids = [self.tokenizer(code, return_tensors='pt', padding='max_length', max_length=512).input_ids.squeeze(0)
                      for code in code_instances]
        
        # Get embeddings from the encoder for each code instance
        embeddings = []
        for code in code_instances:
            inputs = self.tokenizer(code, return_tensors="pt", padding=True, truncation=True).to(self.device)
            with torch.no_grad():
                embedding = self.encoder_model.get_embeddings(inputs)
            embeddings.append(embedding.squeeze(0))
        
        return {
            'encoder_embeddings': torch.stack(embeddings),  # Shape: [4, embedding_size]
            'target_ids': torch.stack(target_ids),           # Shape: [4, max_length]
        }

def create_finetuned_decoder_dataloader(dataframe, encoder_model, tokenizer, device, batch_size=8, shuffle=True):
    dataset = DecoderFineTuneDataset(dataframe, encoder_model, tokenizer, device)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def read_seq_data_with_filter(configs: dict, filterset: pd.DataFrame) -> pd.DataFrame:
    dataset = pd.read_pickle(configs['seq_data_path'])

    allowed_problemIDs = configs['allowed_problem_list']
    dataset = dataset[dataset['problemID'].isin(allowed_problemIDs)]

    allowed_students = pd.unique(filterset[['studentID_1','studentID_2']].values.ravel())

    dataset = dataset[dataset['studentID'].isin(allowed_students)]

    return dataset