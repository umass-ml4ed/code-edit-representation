import os
from datetime import datetime
import torch.optim as optim
import transformers
import neptune
from neptune.utils import stringify_unsupported
import hydra
from omegaconf import OmegaConf
import sys

from data_loader import *
from model import *
from trainer import *
from utils import *
from eval import *
import torch.nn.functional as F

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from itertools import zip_longest

import torch
from tqdm import tqdm

def printCode(code1, code2):
    # Split each code snippet by lines
    code1_lines = code1.splitlines()
    code2_lines = code2.splitlines()

    # Set the column width for each snippet
    col_width = 100

    # Function to wrap text to fit within the column width
    def wrap_text(text, width):
        return [text[i:i+width] for i in range(0, len(text), width)]

    # Wrap each line in both code snippets
    code1_wrapped = [line_part for line in code1_lines for line_part in wrap_text(line, col_width-5)]
    code2_wrapped = [line_part for line in code2_lines for line_part in wrap_text(line, col_width-5)]

    # # Print headers
    # print(f"{'Code Snippet 1':<{col_width}}{'Code Snippet 2':<{col_width}}")
    # print("=" * (col_width * 2))

    # Print wrapped lines side by side
    for line1, line2 in zip_longest(code1_wrapped, code2_wrapped, fillvalue=""):
        print(f"{line1:<{col_width}}{line2:<{col_width}}")


def get_latent_states(dataset, model):
    model.eval()
    res = np.empty((0,128))
    labels = []
    with torch.no_grad():
        for index, row in dataset.iterrows():
            A1 = row['code_i_1']
            A2 = row['code_j_1']
            B1 = row['code_i_2']
            B2 = row['code_j_2']
            
            Da, Db = model([A1, A2, B1, B2])
            Da = Da.cpu().detach().numpy()
            Db = Db.cpu().detach().numpy()
            res = np.concatenate((res, Da), axis=0)
            res = np.concatenate((res, Db), axis=0)
            labels.append(int(row['problemID']))
            labels.append(int(row['problemID']))
    return res, labels

def plot_clusters(dataloader, model, epoch, plotname, run):
    # X represents your high-dimensional data
        X, labels = get_latent_states(dataloader, model)
        tsne = TSNE(n_components=2, random_state=0)
        X_2d = tsne.fit_transform(X)
        plt.close()
        plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels)  # 'labels' should be your true or predicted labels
        plt.title('Epoch ' + str(epoch))
        plt.savefig(plotname + '.png')
        run['clusters/'+plotname].upload(plotname + '.png')

def get_all_embeddings(model, dataloader):
    model.eval()
    embeddings = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating Embeddings", leave=False):  # Assuming you have a dataloader for your dataset
            inputs = batch['inputs']
            labels = batch['labels']
            masks = batch['masks']
            batch_size = labels.shape[0]
            A1 = inputs[:batch_size]
            A2 = inputs[batch_size:2 * batch_size]
            B1 = inputs[2 * batch_size:3 * batch_size]
            B2 = inputs[3 * batch_size:]
            A1_mask = masks[:batch_size]
            A2_mask = masks[batch_size:2 * batch_size]
            B1_mask = masks[2 * batch_size:3 * batch_size]
            B2_mask = masks[3 * batch_size:]

            Da_fc, Db_fc = model(inputs)
            Da_fc = Da_fc.cpu().numpy()
            Db_fc = Db_fc.cpu().numpy()

            A_emb = list(zip(Da_fc, A1, A2, A1_mask, A2_mask))
            B_emb = list(zip(Db_fc, B1, B2, B1_mask, B2_mask))
            embeddings = embeddings + A_emb
            embeddings = embeddings + B_emb
    return embeddings

def removeSpace(text):
    text_no_whitespace = text.replace(" ", "").replace("\n", "").replace("\t", "").replace("\r","")
    return text_no_whitespace


def print_clusters(embeddings, sensitivity, printCode = True, printMask = True):
    embedding_vectors = torch.stack([torch.tensor(embed[0]) for embed in embeddings])
    
    # Step 1: Calculate pairwise distances using torch.cdist for vectorized distance calculation
    distance_matrix = torch.cdist(embedding_vectors, embedding_vectors)
    
    # Step 2: Create a mask for distances within sensitivity
    mask_sensitivity = (distance_matrix < sensitivity)
    
    # Step 3: Iterate through each embedding to check and print clusters
    interestingClusters = 0
    totalClusters = 0
    for i in tqdm(range(len(embeddings)), desc='Printing Clusters', leave=False):
        A1 = removeSpace(embeddings[i][1])
        A2 = removeSpace(embeddings[i][2])
        
        clustercount = 0
        firstequal = 0
        secondequal = 0
        bothequal = 0
        
        # Step 4: Find valid pairs based on mask_sensitivity
        valid_indices = mask_sensitivity[i].nonzero(as_tuple=True)[0]

        for pos in valid_indices:
            if pos > i:  # Ensures each pair is checked only once and avoids self-pairing
                B1 = removeSpace(embeddings[pos][1])
                B2 = removeSpace(embeddings[pos][2])
                
                # Check uniqueness conditions for clusters
                if A1 != B1 and A2 != B2 and A1 != A2 and B1 != B2:
                    clustercount += 1
                    if embeddings[i][3] == embeddings[pos][3]: 
                        firstequal += 1
                    if embeddings[i][4] == embeddings[pos][4]: 
                        secondequal += 1
                    if embeddings[i][3] == embeddings[pos][3] and embeddings[i][4] == embeddings[pos][4]: 
                        bothequal += 1
                        continue

                    if printMask: print(embeddings[pos][3], embeddings[pos][4])
                    if printCode: 
                        printCode(embeddings[pos][1], embeddings[pos][2])
                        print('-' * 119)
        
        # Print the clustercount separator after each primary embedding's clusters are printed
        if(clustercount > 0):
            totalClusters += 1
            if A1 != A2:
                if printMask: print(embeddings[i][3], embeddings[i][4])
                if printCode: printCode(embeddings[i][1], embeddings[i][2])
            print(f"{clustercount}, {firstequal}, {secondequal}, {bothequal}{'*' * 119}")
            print()
            print()
            if clustercount == bothequal: interestingClusters += 1
    print(totalClusters, interestingClusters)




def print_closest(embeddings, configs, sensitivity):
    # Step 1: Extract and convert all embeddings to a tensor for vectorized computation
    embedding_vectors = torch.stack([torch.tensor(embed[0]) for embed in embeddings])
    
    # Step 2: Calculate pairwise distances using torch.cdist for efficient distance calculation
    distance_matrix = torch.cdist(embedding_vectors, embedding_vectors)
    
    # Step 3: Create a mask for distances within configs.margin and sensitivity
    mask_margin = (distance_matrix < configs.margin)
    mask_sensitivity = (distance_matrix < sensitivity)
    
    # Step 4: Track clusters
    clustercount = 0
    n = len(embeddings)
    
    # Step 5: Iterate only over pairs that satisfy the conditions
    for i in tqdm(range(n), desc='Printing Closest', leave=False):
        valid_indices = (mask_margin[i] & mask_sensitivity[i]).nonzero(as_tuple=True)[0]
        mindist, pos = configs.margin, -1
        
        for j in valid_indices:
            if j > i:  # To avoid re-checking pairs and self-pairing
                dist = distance_matrix[i, j].item()
                if dist < mindist:
                    mindist = dist
                    pos = j
        
        # Perform clustering checks
        if pos != -1:
            A1 = removeSpace(embeddings[i][1])
            A2 = removeSpace(embeddings[i][2])
            B1 = removeSpace(embeddings[pos][1])
            B2 = removeSpace(embeddings[pos][2])
            
            if mindist < sensitivity and A1 != B1 and A2 != B2 and A1 != A2 and B1 != B2:
                clustercount += 1
                printCode(embeddings[i][1], embeddings[i][2])
                print('-' * 119)
                printCode(embeddings[pos][1], embeddings[pos][2])
                print('*' * 119)
                print()
                print()
                
    print(len(embeddings), clustercount)



@hydra.main(version_base=None, config_path=".", config_name="configs_cer")
def main(configs):
    now = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Make reproducible
    set_random_seed(configs.seed)    

    # Set device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')  
    print('Current Device: ' + str(device))

    # Initialize the model
    # model = CustomCERModel(configs=configs, device=device)
    _,tokenizer = create_cer_model(configs, device)

    # Path to the checkpoint
    # checkpoint_path = 'checkpoints/20241021_174314' # allowed_problem_list: ['12', '17', '21'] # only if else related problems
    # checkpoint_path = 'checkpoints/20241021_200242' #allowed_problem_list: ['34', '39', '40'] # string problems requiring loops
    # checkpoint_path = 'checkpoints/20241028_201125' # allowed_problem_list: ['46', '71'] # array problems requiring loops
    # checkpoint_path = 'checkpoints/20241029_134451' #all problems
    checkpoint_path = 'checkpoints/20241030_163548' #random (epoch 2) all problem

    # Load the model's state_dict from the checkpoint
    # model.load_state_dict(torch.load(checkpoint_path + '/model', map_location=device))
    model = torch.load(checkpoint_path + '/model')

    train_set = torch.load(checkpoint_path + '/train_set')
    test_set = torch.load(checkpoint_path + '/test_set')
    valid_set = torch.load(checkpoint_path + '/valid_set')
    
    ## load data
    collate_fn = CollateForCER(tokenizer=tokenizer, configs=configs, device=device)
    train_loader = make_dataloader(train_set, collate_fn=collate_fn, configs=configs)
    valid_loader = make_dataloader(valid_set, collate_fn=collate_fn, configs=configs)
    test_loader  = make_dataloader(test_set , collate_fn=collate_fn, configs=configs)
    

    test_embeddings = get_all_embeddings(model, test_loader)

    # print(test_embeddings)
    # print_closest(test_embeddings, configs, configs.margin / 100)
    print_clusters(embeddings=test_embeddings, sensitivity=configs.margin / 100, printCode=False, printMask=False)

if __name__ == "__main__":
    main()
