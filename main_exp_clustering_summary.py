import os
from datetime import datetime
import torch.optim as optim
import transformers
import neptune
from neptune.utils import stringify_unsupported
import hydra
from omegaconf import OmegaConf
import sys
import numpy as np

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
import seaborn as sns

from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers.modeling_outputs import BaseModelOutput
from datatypes import *

from tqdm import tqdm
# from evaluator.CodeBLEU import calc_code_bleu
from main_exp_gen_code_decoder import *
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from openai import OpenAI

def generate_code_embeddings(model, dataset, tokenizer, configs, device):
    collate_fn = CollateForCER(tokenizer=tokenizer, configs=configs, device=device)
    dataloader = make_dataloader_experiment(dataset, collate_fn=collate_fn, configs=configs)

    model.eval()
    embeddings, code_pairs = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating Embeddings", leave=False):
            tokenized_inputs = tokenizer(batch['inputs'], return_tensors="pt", padding=True, truncation=True).to(device)
            Da, Db = model.get_edit_encodings_tokenized(tokenized_inputs)

            A1, A2, B1, B2 = model.batch_unpack(batch['inputs'], Da.shape[0])
            embeddings.extend([Da.cpu(), Db.cpu()])
            code_pairs += [(a1, a2) for a1, a2 in zip(A1, A2)]
            code_pairs += [(b1, b2) for b1, b2 in zip(B1, B2)]
    
    return np.vstack([e.numpy() for e in embeddings]), code_pairs

def generate_cluster_summary(model, dataset, tokenizer, configs, device):
    embeddings, code_pairs = generate_code_embeddings(model, dataset, tokenizer, configs, device)
    
    # dbscan = DBSCAN(eps=configs.margin , min_samples=1, metric='euclidean')
    # cluster_labels = dbscan.fit_predict(embeddings)
    
    num_clusters = 100  # Define the number of clusters
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings)

    print(f"Clusters found: {len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)}. Total instances: {len(embeddings)}")
    
    # pca = PCA(n_components=2)
    # reduced_embeddings = pca.fit_transform(embeddings)

    tsne = TSNE(n_components=2, random_state=0)
    reduced_embeddings = tsne.fit_transform(embeddings)

    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=cluster_labels, cmap='viridis', alpha=0.7)
    # plt.title('Clustering Summary')
    # plt.colorbar()
    plt.savefig('cluster_summary.png')

    summarize_clusters_with_gpt4(cluster_labels=cluster_labels, code_pairs=code_pairs)

    return cluster_labels, code_pairs

def summarize_clusters_with_gpt4(cluster_labels, code_pairs):

    # Ensure the API key is set in your environment or explicitly provide it
    api_key = os.getenv("OPENAI_API_KEY")  # Fetch from environment variable
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set.")

    # Initialize the OpenAI client with the API key
    client = OpenAI(api_key=api_key)

    clusters = {}
    for code, label in zip(code_pairs, cluster_labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(code)
    
    summaries = {}
    newdatalist = []
    for label, samples in clusters.items():
        sampled_codes = samples[:5]
        newdata = []
        newdata.append(label)
        # prompt = f"Provide a single brief summary of edits for all the provided code pairs. The summary should be a unified one across different problems. Focus on common edit patterns between code pairs.\n"
        prompt = "You are given the following 5 code pairs. Each of them belong to a possibly different student. The 'Inital Code' is student's first attempt and the 'Next Code' is their next attempt to correct 'Initial Code'. Analyze these and produce the common mistakes in the 'Initial Code's. Also find common debugging patterns the students used to get to 'Next Code'. Focus on finding common patterns that exist in all 5.\n"
        for (a,b) in sampled_codes:
            newdata.append(a)
            newdata.append(b)
            prompt += 'Initial Code:\n' + a
            prompt += 'Next Code:\n' + b + '\n\n'

        prompt += "Make sure the output is just one paragraph and very brief."
        print(prompt)
        response = client.chat.completions.create(
            # model="gpt-4o-mini", 
            model='gpt-4o',
            messages=[
                {"role": "system", "content": "You are a coding assistant."}, {"role": "user", "content": prompt}
                ],
            max_tokens=512,
            temperature=0.7
        )
        summaries[label] = response.choices[0].message.content
        newdata.append(summaries[label])
        newdatalist.append(newdata)
        print()
        print(summaries[label])
        print('--------------------------------------------------------------------')
        sys.stdout.flush()  # Forces the buffer to write to stdout
        # break

    with open('cluster_summaries.txt', 'w') as f:
        for label, summary in summaries.items():
            f.write(f"Cluster {label}:\n{summary}\n\n")
    
    newdf = pd.DataFrame(newdatalist, columns=['clusterID', 
                                           'code_i_1', 'code_j_1',
                                           'code_i_2', 'code_j_2',
                                           'code_i_3', 'code_j_3',
                                           'code_i_4', 'code_j_4',
                                           'code_i_5', 'code_j_5',
                                           'summary'
                                           ])
    newdf.to_csv('gpt4o_cluster_summaries.csv', index=False )


@hydra.main(version_base=None, config_path=".", config_name="configs_cer")
def main(configs):
    now = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Make reproducible
    set_random_seed(configs.seed)

    # Set device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('Current Device: ' + str(device))

    # Initialize the model
    tokenizer = create_tokenizer(configs)
    checkpoint_path = configs.model_save_dir

    data_checkpoint_name = '20250130_212344' #cerd, all, reconstruction =.5
    _, train_set, valid_set, test_set = load_checkpoint_model_and_data(checkpoint_name=data_checkpoint_name, configs=configs) #to keep the data constant over experiments
    
    # Path to the checkpoint
    # checkpoint_name = '20241209_165650' # with regularization, if else  
    # checkpoint_name = '20241209_194800' # with regularization, if else, exclusive problems between train and test
    # checkpoint_name = '20241211_195813' #with reg, student split, all problems.
    # checkpoint_name = '20241213_224930' #with reg, student split, all problems. higher reconstruction lambda
    # checkpoint_name = '20241214_000113' #with reg, student split, all problems. t5-large
    # checkpoint_name = '20241215_192723' #with reg, student split, all problems. reconstruction lambda = 1.5
    # checkpoint_name = '20241216_192316' #with reg, student split, all problems. reconstruction lambda = 2. t5-base
    # checkpoint_name = '20241217_212527' #with reg, student split, all problems. reconstruction lambda = 2. code-t5-base

    # checkpoint_name = '20250130_211733' #cerdd, all, reconstruction =.5
    # checkpoint_name = '20250130_212046' #cerdd, all, reconstruction = 1
    # checkpoint_name = '20250130_212102' #cerdd, all, reconstruction = 1.5
    # checkpoint_name = '20250130_212215' #cerdd, all, reconstruction = 2
    # checkpoint_name = '20250130_212223' #cerdd, all, reconstruction = 3

    # checkpoint_name = '20250130_212344' #cerd, all, reconstruction =.5
    # checkpoint_name = '20250130_212343' #cerd, all, reconstruction = 1
    # checkpoint_name = '20250130_213807' #cerd, all, reconstruction = 1.5
    # checkpoint_name = '20250130_215832' #cerd, all, reconstruction = 2
    # checkpoint_name = '20250130_220007' #cerd, all, reconstruction = 3
    # checkpoint_name = '20250208_162240' #cerd, all, reconstruction = 4
    # checkpoint_name = '20250208_162301' #cerd, all, reconstruction = 5

    # checkpoint_name = '20250206_190729' #cerd, all, reconstruction = 3, regularization = 2

    # checkpoint_name = '20250211_212450' #cerd, all, reconstruction = 2, codet5-large
    # checkpoint_name = '20250211_212856' #cerd, all, reconstruction = 3, codet5-large
    # checkpoint_name = '20250211_212656' #cerdd, all, reconstruction = 2, codet5-large
    # checkpoint_name = '20250211_213144' #cerdd, all, reconstruction = 3, codet5-large

    checkpoint_name = '20250215_160225' #cerd, all, reconstruction = 2, codet5-base, rec and trans
    # checkpoint_name = '20250215_160712' #cerdd, all, reconstruction = 2, codet5-base, rec and trans

    # checkpoint_name = '20250216_021903' #cerd, all, recstruction = 1, contrastive = 1, regularization = 0
    # checkpoint_name = '20250216_022008' #cerd, all, recstruction = 0, contrastive = 1, regularization = 1
    # checkpoint_name = '20250216_022136' #cerd, all, recstruction = 1, contrastive = 0, regularization = 1

    cerd_model, _, _, _ = load_checkpoint_model_and_data(checkpoint_name=checkpoint_name, configs=configs)

    # Example usage
    generated_summary = generate_cluster_summary(model= cerd_model, dataset=train_set, tokenizer=tokenizer, configs=configs, device=device)

if __name__ == "__main__":
    main()
