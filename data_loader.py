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
    @param configs.label_type: whether to use binarized label, raw label, or ternery label
    @param configs.max_len: maximum allowed length for each student's answer sequence. longer
                    than this number will be truncated and set as new student(s)
    @param configs.seed: reproducibility
    '''
    # load dataset
    dataset = pd.read_pickle(configs.data_path + '/dataset.pkl')

    #split the dataset into two, one for is_similar = True and the other for is_similar = False
    dataset_true = dataset[dataset['is_similar'] == True]
    dataset_false = dataset[dataset['is_similar'] == False]

    #sample the dataset_false to have the same number of rows as dataset_true
    dataset_false = dataset_false.sample(n=dataset_true.shape[0])

    #concatenate the two datasets
    dataset = pd.concat([dataset_true, dataset_false])
        
    ## if only testing, subsample part of dataset
    if configs.testing:
        dataset = dataset.sample(n=120)
        # Sort sampled dataset for timestep columm creation logic to work below
        # dataset = dataset.sort_values(by=["SubjectID", "AssignmentID", "ProblemID"])

    # # choose label format
    # if configs.label_type == 'binary':
    #     scores_y = []
    #     for item in dataset['Score_y']:
    #         if item >= 2:
    #             scores_y.append(1)
    #         else:
    #             scores_y.append(0)
    #     dataset['Score'] = scores_y
    # elif configs.label_type == 'ternery':
    #     dataset['Score'] = dataset['Score_y']
    # elif configs.label_type == 'raw':
    #     dataset['Score'] = dataset['Score_x']
    # dataset = dataset.drop(columns=['Score_x','Score_y'])
    
    # ## optionally keep only the first answer by the student
    # if configs.first_ast_convertible:
    #     ('only using first ast-convertible code')
    #     dataset = dataset.drop_duplicates(
    #                     subset = ['SubjectID', 'ProblemID'],
    #                     keep = 'first').reset_index(drop = True)
    
    # BUG: Timestep column with time ordering is different from real world data present in column order and column server timestamp
    # We ignore this since we're working in IRT setting ignoring time, for KT the data should be ordered according to real world time ordering
    
    ## Make a new dataframe from each pair of rows from the dataset only where the matcesh_i and matches_j is exacatly equal for the two rows
    ## The columns in each row are pid,sid,matches_i,cid_i,code_i,score_i,score_calc_i,matches_j,cid_j,code_j,score_j,score_calc_j
    ## The new dataframe has the columns pid,sid_1,matches_i_1,cid_i_1,code_i_1,score_i_1,score_calc_i_1,matches_j_1,cid_j_1,code_j_1,score_j_1,score_calc_j_1,sid_2,matches_i_2,cid_i_2,code_i_2,score_i_2,score_calc_i_2,matches_j_2,cid_j_2,code_j_2,score_j_2,score_calc_j_2
    
    ## The new dataframe is then saved as a pickle file

    
    

    


    ## split a student's record into multiples 
    ## if it exceeds configs.max_len, change the subject ID to next one
    # prev_subject_id = 0
    # subjectid_appendix = []
    # timesteps = []
    # for i in tqdm(range(len(dataset)), desc="splitting students' records ..."):
    #     if prev_subject_id != dataset.iloc[i].SubjectID:
    #         # when encountering a new student ID
    #         prev_subject_id = dataset.iloc[i].SubjectID
    #         accumulated = 0
    #         id_appendix = 1
    #     else:
    #         accumulated += 1
    #         if accumulated >= configs.max_len:
    #             id_appendix += 1
    #             accumulated = 0
    #     timesteps.append(accumulated)
    #     subjectid_appendix.append(id_appendix)
    # dataset['timestep'] = timesteps
    # dataset['SubjectID_appendix'] = subjectid_appendix
    # dataset['SubjectID'] = [dataset.iloc[i].SubjectID + \
    #             '_{}'.format(dataset.iloc[i].SubjectID_appendix) for i in range(len(dataset))]

    ## Each subject ID implies a student
    # students = dataset['SubjectID'].unique()

    # Train, val, test split 
    # Keep copy of dataset with timestep=0 for creating LSTM input dataset since we require (p_0, c_0) to compute h_0 used to predict c_1
    # dropped_dataset = dataset.copy()
    # Drop entries with timestep=0 since we don't have student history (p_i, c_i) to compute student knowledge state to predict c_0 for p_0
    # dropped_dataset = dropped_dataset.drop(dropped_dataset.index[dropped_dataset['timestep'] == 0]).reset_index(drop = True)
    # Split on entries instead of on students
    trainset, testset = train_test_split(dataset, test_size=configs.test_size, random_state=configs.seed)
    validset, testset = train_test_split(testset, test_size=0.5, random_state=configs.seed)

    return trainset, validset, testset, dataset#, students


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
#     okt_dataset = []
#     students = dataset_split['SubjectID'].unique()
#     for student in students:
#         subset = dataset_split[dataset_split.SubjectID==student]
#         for t in range(len(subset)):
#             # Set step = timestep-1 for alignment with LSTM input dataset [(p_i, c_i)] to ensure h_t computed using [(p_0, c_0), ..., (p_t, c_t)] is used to predict c_{t+1}
#             okt_dataset.append({
#                 'SubjectID': student,
#                 'ProblemID': subset.iloc[t].ProblemID,
#                 'step': subset.iloc[t].timestep-1, 
#                 'next_Score': subset.iloc[t].Score,
#                 'next_prompt': subset.iloc[t].prompt,
#                 'next_code': subset.iloc[t].Code,
#             })
#     del dataset_split
    
#     # dictionary, key=student id, value=list of lstm inputs at each time step
#     if do_lstm_dataset:
#         lstm_dataset = {}
#         students = dataset_full['SubjectID'].unique()
#         for student in students:
#             lstm_dataset[student]=dataset_full[dataset_full.SubjectID==student].input.tolist()
#         del dataset_full
#         return okt_dataset, lstm_dataset
#     else:
#         return okt_dataset


# def make_dataloader(dataset_split, dataset_full, collate_fn, configs, n_workers=0, do_lstm_dataset=True, train=True):
def make_dataloader(dataset: pd.DataFrame, collate_fn: Callable, configs: dict, n_workers: int = 0, train: bool = True) -> torch.utils.data.DataLoader:
    # Make two datasets: one with a list of dict (for GPT), and another a dict with student_id as key (for LSTM to compute knowledge states)
    shuffle = True if train else False
    # if do_lstm_dataset:
    #     okt_dataset, lstm_dataset = make_pytorch_dataset(dataset_split, dataset_full, do_lstm_dataset)
    #     data_loader = torch.utils.data.DataLoader(
    #         okt_dataset, collate_fn=collate_fn, shuffle=shuffle, batch_size=configs.batch_size, num_workers=n_workers)   
    #     return okt_dataset, data_loader, lstm_dataset
    # else:
    #     okt_dataset = make_pytorch_dataset(dataset_split, dataset_full, do_lstm_dataset)
    #     data_loader = torch.utils.data.DataLoader(
    #         okt_dataset, collate_fn=collate_fn, shuffle=shuffle, batch_size=configs.batch_size, num_workers=n_workers)   
    #     return okt_dataset, data_loader
    # print('dataset')
    # print(dataset)
    dataset = make_pytorch_dataset(dataset=dataset)
    # data_loader = torch.utils.data.DataLoader(dataset, collate_fn=collate_fn, shuffle=shuffle, batch_size=configs.batch_size, num_workers=n_workers)
    data_loader = torch.utils.data.DataLoader(dataset, shuffle=shuffle, batch_size=configs.batch_size, num_workers=n_workers)
    return data_loader
    

# def get_lstm_inputs(configs, tokenizer, student_id_to_index, train_set, dataset, collate_fn):
    
#     _, _, lstm_inputs = make_dataloader(train_set, dataset, 
#                                                    collate_fn=collate_fn, 
#                                                    configs=configs, do_lstm_dataset=True)
    
#     return lstm_inputs


# def build_input_with_special_tokens(prompt, code, tokenizer):
#     # Match GPT2 pretraining input style: https://github.com/huggingface/transformers/issues/3311
#     # https://huggingface.co/patrickvonplaten/bert2gpt2-cnn_dailymail-fp16#training-script
#     # Input format: <|endoftext|>question: <question> student written code: <code><|endoftext|>
#     # Start completion (student code) with whitespace
#     input = build_prompt_with_special_tokens(prompt, tokenizer) + " " + code.strip() + tokenizer.eos_token

#     return input


# def build_prompt_with_special_tokens(prompt, tokenizer):
#     # Remove delimiter : in prompt since we use it to calculate prompt length
#     if( ":" in prompt ):
#         prompt = prompt.replace(":", "")
#     # Phrase "student written code:" should serve as our separator between prompt and completion
#     assert "student written code" not in prompt
#     prompt = tokenizer.bos_token + "question: " + prompt + " student written code:"

#     return prompt


class CollateForCER(object):
    def __init__(self, tokenizer: tokenizer, configs: dict, device: torch.device):
        self.tokenizer = tokenizer
        # # Pad if required with <|endoftext|> tokens on the right of input since GPT2 uses absolute position embeddings
        # assert self.tokenizer.padding_side == "right"
        self.configs = configs
        # self.student_id_to_index = student_id_to_index
        self.device = device
        # # Token id 25 corresponds to ":" in vocab https://huggingface.co/gpt2/raw/main/vocab.json
        # self.delimiter_token_id = 25
        

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

        # inputs = torch.stack([torch.tensor([self.tokenizer(b['A1'], return_tensors='pt', padding=True, truncation=True),
        #            self.tokenizer(b['A2'], return_tensors='pt', padding=True, truncation=True),
        #            self.tokenizer(b['B1'], return_tensors='pt', padding=True, truncation=True),
        #            self.tokenizer(b['B2'], return_tensors='pt', padding=True, truncation=True)]) for b in batch])
        # # inputs = [(b['A1'], b['A2'], b['B1'], b['B2']) for b in batch]
        # # inputs = self.tokenizer(inputs, return_tensors='pt', padding=True, truncation=True)
        # # inputs = torch.tensor(inputs).to(self.device)
        # labels = torch.stack([torch.tensor(b['label'], dtype=torch.long) for b in batch])
        # # labels = torch.tensor(labels).to(self.device)
        # return {'input': inputs, 'label': labels}

        # inputs_text = [build_input_with_special_tokens(b['next_prompt'], b['next_code'], self.tokenizer) for b in batch]
        # inputs = self.tokenizer(inputs_text, return_tensors='pt', padding=True, truncation=True)
        # inputs_ids, attention_mask = inputs['input_ids'].to(self.device), inputs['attention_mask'].to(self.device)

        # # Handle truncation: Replace last token id with tokenizer.eos_token_id to ensure generation ends with eos_token_id
        # inputs_ids[:, -1] = self.tokenizer.eos_token_id

        # # Find prompt length which is needed to linearly combine student knowledge state with prompt tokens only
        # # To find prompt length we find the second occurence of delimiter ":" in <|endoftext|>question: <question> student written code: <code><|endoftext|>
        # delimiter_indices = torch.where(inputs_ids == self.delimiter_token_id, 1, 0)
        # # Ignore first occurence of delimiter at index 2 since our prompt always starts with <|endoftext|>question:
        # delimiter_indices[:, 2] = 0
        # # Argmax returns first occurence of maximum value. Here the first occurence of maximum value will be the second occurence of delimiter (we ignored the first occurence)
        # prompt_id_lens = torch.argmax(delimiter_indices, dim=-1)
        # # Add 1 since length = zero-based index + 1
        # prompt_id_lens = torch.add(prompt_id_lens, 1)
        
        # # Compute labels
        # labels = inputs_ids.detach().clone()
        # # Ignore padding
        # labels = labels.masked_fill((attention_mask == 0), -100)
        # # Use only code tokens, ignore prompt tokens
        # range_tensor = torch.arange(inputs_ids.size(1), device=self.device).unsqueeze(0)
        # range_tensor = range_tensor.repeat(prompt_id_lens.size(0), 1)
        # mask_tensor = (range_tensor < prompt_id_lens.unsqueeze(-1)) 
        # labels[mask_tensor] = -100
        
        # students = [b['SubjectID'] for b in batch]
        # # List of lists instead of flat list to match gather function indexing call in model
        # student_ids = torch.LongTensor( [self.student_id_to_index[b['SubjectID']] for b in batch] ).to(self.device)
        # timesteps = [b['step'] for b in batch]

        # """
        # # Print sample batch
        # print("Sample batch:")
        # for ids in inputs_ids:
        #     print(self.tokenizer.decode(ids))
        # print("Input ids:", inputs_ids)
        # print("Attention mask:", attention_mask)
        # print("Labels:", labels)
        # """

        # return inputs_ids, attention_mask, labels, prompt_id_lens, students, timesteps, student_ids