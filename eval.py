import torch
import pickle
import torch.nn.functional as F
import nltk
from nltk import ngrams
import os
from neptune.utils import stringify_unsupported
from multiprocessing import Pool
from nltk.translate.bleu_score import SmoothingFunction
import abc
from tqdm import tqdm
from pdb import set_trace
import hydra

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

def get_accuracy(model, input_set, configs):
    # model.eval()
    correct = 0
    total = 0
    i = 0
    target_list = []
    predicted_list = []
    dist_list = []
    with torch.no_grad():
        for index, row in input_set.iterrows():
            A1 = row['code_i_1']
            A2 = row['code_i_2']
            B1 = row['code_j_1']
            B2 = row['code_j_2']
            inputs = [A1, A2, B1, B2]
            target = row['is_similar']
            outputs = model.forward(inputs)
            # outputs = model.forward(A1, A2, B1, B2)
            # output = torch.round(torch.sigmoid(output))
            output1, output2 = outputs
            # print(output1, output2)
            dist = F.pairwise_distance(output1, output2)
            if dist < configs.margin:
                predicted = 1
            else:
                predicted = 0
            target_list.append(target)
            predicted_list.append(predicted)
            dist_list.append(dist.cpu().item())
            total += 1
            if (target == True and predicted == 1) or (target == False and predicted == 0):
                correct += 1
            # if i % 500 == 0: print(correct, total, predicted, target)
            i += 1
    print(target_list)
    print(predicted_list)
    dist_list = [f"{dist:.3f}" for dist in dist_list]
    print(dist_list)
    return correct / total


def get_model_accuracy(configs, model, train_set, test_set, valid_set):
    train_accuracy = get_accuracy(model, train_set, configs)
    print(f'Train accuracy: {train_accuracy}')
    test_accuracy = get_accuracy(model, test_set, configs)
    print(f'Test accuracy: {test_accuracy}')
    valid_accuracy = get_accuracy(model, valid_set, configs)
    print(f'Valid accuracy: {valid_accuracy}')

    return train_accuracy, test_accuracy, valid_accuracy


# if __name__ == "__main__":
#     #torch.set_printoptions(profile="full")
#     main()