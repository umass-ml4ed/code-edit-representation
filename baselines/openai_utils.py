import os
from openai import OpenAI
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

import sys
sys.path.append("..")
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

# Ensure the API key is set in your environment or explicitly provide it
api_key = os.getenv("OPENAI_API_KEY")  # Fetch from environment variable
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set.")

# Initialize the OpenAI client with the API key
client = OpenAI(api_key=api_key)


# Function to generate edit representation
def generate_edit_representation(original_code, modified_code):
    """
    Step 1: Generate a textual representation of the edits.
    """
    prompt = (
        f"The original code snippet is:\n{original_code}\n\n"
        f"The modified code snippet is:\n{modified_code}\n\n"
        "Describe the edits required to transform the original code into the modified code. "
        # "Make sure the edit description is not more than 5 lines."
        "Make it as generalized as possible."
        # "Provide a structured and detailed explanation of the changes."
    )
    
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a coding assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=512,
        temperature=0.7
    )
    
    edit_representation = completion.choices[0].message.content
    return edit_representation

# Function to reproduce modified code
def reproduce_modified_code(original_code, edit_representation):
    """
    Step 2: Generate the modified code from the original code and edit description.
    """
    prompt = (
        f"The original code snippet is:\n{original_code}\n\n"
        f"The description of the edits is:\n{edit_representation}\n\n"
        "Apply the described edits to the original code and provide only the modified code. Write nothing else. Not even the programming language and markdown tags."
    )
    
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a coding assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=512,
        temperature=0.7
    )
    
    reproduced_code = completion.choices[0].message.content
    return reproduced_code

