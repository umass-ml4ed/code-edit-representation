from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
import pandas as pd
from openai_utils import *

# Function to cluster code pairs based on GPT-4 labels
def cluster_code_pairs(dataset, n_clusters=5):
    # Step 1: Generate labels using GPT-4
    labels = []
    for iter, (index, row) in enumerate(tqdm(dataset.iterrows(), desc="Generating GPT4 Labels", leave=False)):
        a1, a2 = row['code_i_1'], row['code_j_1']
        label = generate_edit_representation(a1, a2)
        labels.append(label)

        # b1, b2 = row['code_i_2'], row['code_j_2']
        # label = generate_edit_representation(b1, b2)
        # labels.append(label)

        # if iter == 3:
        #     break
    
    # Step 2: Encode labels using SentenceTransformers or another embedding model
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Use a lightweight transformer model
    label_embeddings = model.encode(labels, show_progress_bar=True)
    
    # Step 3: Perform clustering
    clustering_model = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = clustering_model.fit_predict(label_embeddings)
    
    # Step 4: Add cluster labels to the dataset
    dataset['cluster'] = cluster_labels
    return dataset, clustering_model

@hydra.main(version_base=None, config_path="..", config_name="configs_cer")
def main(configs):
    # Path to the checkpoint
    checkpoint_name = '20241209_165650' # with regularization, if else  
    # checkpoint_name = '20241209_194800' # with regularization, if else, exclusive problems between train and test
    # checkpoint_name = '20241211_195813' #with reg, student split, all problems.
    # checkpoint_name = '20241213_224930' #with reg, student split, all problems. higher reconstruction lambda
    # checkpoint_name = '20241214_000113' #with reg, student split, all problems. t5-large
    # checkpoint_name = '20241215_192723' #with reg, student split, all problems. reconstruction lambda = 1.5
    # checkpoint_name = '20241216_192316' #with reg, student split, all problems. reconstruction lambda = 2. t5-base
    # checkpoint_name = '20241217_212527' #with reg, student split, all problems. reconstruction lambda = 2. code-t5-base

    # Load the dataset
    _, _, _, test_set = load_checkpoint_model_and_data(checkpoint_name=checkpoint_name, configs=configs)
    
    # Perform clustering
    clustered_dataset, clustering_model = cluster_code_pairs(test_set)
    
    # Analyze the clusters
    print(clustered_dataset.groupby('cluster').size())
    
    # Save the clustered dataset for further analysis
    clustered_dataset.to_csv("clustered_code_pairs.csv", index=False)

if __name__ == "__main__":
    main()
