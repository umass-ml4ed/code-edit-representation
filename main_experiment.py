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

import torch
from tqdm import tqdm
import seaborn as sns



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
                        printCodePairSideBySide(embeddings[pos][1], embeddings[pos][2])
                        print('-' * 119)
        
        # Print the clustercount separator after each primary embedding's clusters are printed
        if(clustercount > 0):
            totalClusters += 1
            if A1 != A2:
                if printMask: print(embeddings[i][3], embeddings[i][4])
                if printCode: printCodePairSideBySide(embeddings[i][1], embeddings[i][2])
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
    diffclustercount = 0
    # Step 5: Iterate only over pairs that satisfy the conditions
    for i in tqdm(range(n), desc='Printing Closest', leave=False):
        valid_indices = (mask_margin[i] & mask_sensitivity[i]).nonzero(as_tuple=True)[0]
        mindist, pos = configs.margin, -1
        A1_mask = embeddings[i][3]
        A2_mask = embeddings[i][4]
        
        for j in valid_indices:
            if j > i:  # To avoid re-checking pairs and self-pairing
                B1_mask = embeddings[j][3]
                B2_mask = embeddings[j][4]

                if A1_mask == B1_mask or A2_mask == B2_mask: continue # only process different problems in the same cluster
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
                printCodePairSideBySide(embeddings[i][1], embeddings[i][2])
                print('-' * 119)
                printCodePairSideBySide(embeddings[pos][1], embeddings[pos][2])
                print('*' * 119)
                print()
                print()

                
    print(len(embeddings), clustercount)


def calculate_cluster_diameters(embeddings, printCode=True, printMask=True):
    embedding_vectors = torch.stack([torch.tensor(embed[0]) for embed in embeddings])
    
    # Step 1: Calculate pairwise distances using torch.cdist for vectorized distance calculation
    distance_matrix = torch.cdist(embedding_vectors, embedding_vectors)
    
    # Step 2: Create verdict masks and group embeddings by verdict_mask
    verdict_masks = [embed[3]+embed[4] for embed in embeddings]
    clusters = {}
    
    for idx, verdict_mask in enumerate(verdict_masks):
        mask_key = verdict_mask#.cpu().numpy().tobytes()  # Hashable key for each unique mask
        if mask_key not in clusters:
            clusters[mask_key] = []
        clusters[mask_key].append(idx)
    
    diameters = []
    # Step 3: Iterate through clusters and calculate the maximum distance (diameter) within each
    for mask_key, indices in tqdm(clusters.items(), desc='Calculating Cluster Diameters', leave=False):
        if len(indices) < 2:
            continue  # Skip clusters with fewer than 2 elements, as they have no diameter
        
        # Extract distances for all pairs within the current cluster
        cluster_distances = distance_matrix[indices][:, indices]
        
        # Calculate the maximum distance within the cluster
        max_distance = cluster_distances.max().item()
        print(max_distance)
        diameters.append(max_distance)  # Collect for histogram
    return diameters

def plot_diameters(diameter_optimal, diameter_random, name):
    # Plot histogram of cluster diameters with more bins and a log scale for better visibility of smaller bars
    plt.figure(figsize=(10, 6))
    
    # Histogram for optimal diameters with log scale on y-axis
    plt.hist(diameter_random, bins=30, color='red', edgecolor='black', alpha=0.3, label='Initial')
    plt.hist(diameter_optimal, bins=30, color='blue', edgecolor='black', alpha=0.5, label='Optimal')

    # Log scale on the y-axis
    plt.yscale('log')
    plt.ylim(1, 10000)
    # Labels, title, and legend
    plt.xlabel('Cluster Diameter (Max Distance)')
    plt.ylabel('Frequency (Log Scale)')
    plt.title('Distribution of Cluster Diameters with Log Scale')
    plt.legend(loc='upper right')
    
    # Save the plot as an image file
    plt.savefig(name)
    plt.show()

from tqdm import tqdm
import torch

def calculate_centroid_distances(embeddings):
    embedding_vectors = torch.stack([torch.tensor(embed[0]) for embed in embeddings])
    verdict_masks = [embed[3] + embed[4] for embed in embeddings]
    
    # Group indices by verdict masks and calculate centroids
    clusters = {}
    centroids = {}
    
    for idx, verdict_mask in enumerate(verdict_masks):
        mask_key = verdict_mask
        if mask_key not in clusters:
            clusters[mask_key] = []
        clusters[mask_key].append(idx)
    
    for mask_key, indices in clusters.items():
        # Calculate centroid as the mean of all embeddings in the cluster
        centroids[mask_key] = embedding_vectors[indices].mean(dim=0)
    
    intra_cluster_distances = []
    inter_cluster_distances = []
    
    # Calculate intra-cluster distances from each point to its cluster centroid
    for mask_key, indices in tqdm(clusters.items(), desc="Calculating Intra-Cluster Distances"):
        centroid = centroids[mask_key]
        for idx in indices:
            dist = torch.dist(embedding_vectors[idx], centroid)
            intra_cluster_distances.append(dist.item())
    
    # Calculate inter-cluster distances between centroids
    mask_keys = list(centroids.keys())
    for i in tqdm(range(len(mask_keys)), desc="Calculating Inter-Cluster Distances"):
        for j in range(i + 1, len(mask_keys)):
            dist = torch.dist(centroids[mask_keys[i]], centroids[mask_keys[j]])
            inter_cluster_distances.append(dist.item())
    
    return intra_cluster_distances, inter_cluster_distances


def plot_distance_distributions(intra_mask_distances, inter_mask_distances, name):
    plt.figure(figsize=(12, 6))
    
    # Histogram plot for intra and inter-mask distances
    
    plt.hist(inter_mask_distances, bins=30, color='salmon', edgecolor='black', alpha=0.5, label='Inter-Mask Distances')
    plt.hist(intra_mask_distances, bins=30, color='skyblue', edgecolor='black', alpha=0.7, label='Intra-Mask Distances')

    plt.xlabel('Distance')
    plt.ylabel('Frequency')
    plt.title('Distribution of Intra-Mask and Inter-Mask Distances')
    plt.legend(loc='upper right')
    
    plt.savefig(name + "_hist.png")
    plt.show()

    # Boxplot for intra and inter-mask distances
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=[intra_mask_distances, inter_mask_distances], palette=["skyblue", "salmon"])
    plt.xticks([0, 1], ['Intra-Mask Distances', 'Inter-Mask Distances'])
    plt.ylabel('Distance')
    plt.title('Boxplot of Intra-Mask and Inter-Mask Distances')
    
    plt.savefig(name + "_boxplot.png")
    plt.show()


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
    model0,tokenizer = create_cer_model(configs, device)

    # Path to the checkpoint
    checkpoint_path = configs.model_save_dir
    checkpoint_path += '/20241029_134451' #all problems
    # checkpoint_path += '/20241030_163548' #random (epoch 2) all problem
    # checkpoint_path += '/20241031_190036' #epoch 8, margin 1

    model = torch.load(checkpoint_path + '/model')

    train_set = torch.load(checkpoint_path + '/train_set')
    test_set = torch.load(checkpoint_path + '/test_set')
    valid_set = torch.load(checkpoint_path + '/valid_set')
    
    ## load data
    collate_fn = CollateForCER(tokenizer=tokenizer, configs=configs, device=device)
    train_loader = make_dataloader(train_set, collate_fn=collate_fn, configs=configs)
    valid_loader = make_dataloader(valid_set, collate_fn=collate_fn, configs=configs)
    test_loader  = make_dataloader(test_set , collate_fn=collate_fn, configs=configs)
    

    data_loader = test_loader
    embeddings_optimal = get_all_embeddings(model, data_loader)

    # print(test_embeddings)
    print_closest(embeddings_optimal, configs, configs.margin/100)
    # print_clusters(embeddings=test_embeddings, sensitivity=configs.margin / 100, printCode=False, printMask=False)
    # dia_opt = calculate_cluster_diameters(embeddings=embeddings_optimal)

    # checkpoint_path = 'checkpoints/20241031_190036' #epoch 8, margin 1
    # model = torch.load(checkpoint_path + '/model')
    # # model = model0
    # embeddings_random = get_all_embeddings(model, data_loader)
    # dia_rand = calculate_cluster_diameters(embeddings=embeddings_random)

    # plot_diameters(diameter_optimal=dia_opt, diameter_random=dia_rand, name='Hist_compare_train8.png')

    # embeddings = get_all_embeddings(model=model, dataloader=data_loader)
    # intra_mask_distances, inter_mask_distances = calculate_centroid_distances(embeddings)
    # plot_distance_distributions(intra_mask_distances, inter_mask_distances, "mask_distance_comparison")



if __name__ == "__main__":
    main()
