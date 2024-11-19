import os
import torch
import random
import numpy as np
import yaml
from munch import Munch
from sklearn.metrics import accuracy_score, roc_auc_score
from itertools import zip_longest
import re

def set_random_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

def aggregate_metrics(log):
    results = {}
    for k in log[0].keys():
        if k == 'auc':
            logits = np.concatenate([x[k]['logits'].numpy().reshape(-1) for x in log])
            scores = np.concatenate([x[k]['scores'].numpy().reshape(-1) for x in log])
            results[k] = roc_auc_score(scores, logits)
        elif k == 'pred':
            res = np.concatenate([x[k].numpy().reshape(-1) for x in log])
            results[k] = res.sum()
        else:
            res = np.concatenate([x[k].numpy().reshape(-1) for x in log])
            results[k] = np.mean(res)
    return results

def format_java_code(java_code):
    """
    Basic formatter for Java code. Adds indentation based on braces and adds line breaks.
    """
    # Add new lines after semicolons, braces, and return statements
    java_code = re.sub(r';', ';\n', java_code)
    java_code = re.sub(r'{', '{\n', java_code)
    java_code = re.sub(r'}', '}\n', java_code)
    java_code = re.sub(r'return ', r'\nreturn ', java_code)

    # Trim and split into lines
    lines = java_code.strip().split('\n')
    
    # Manage indentation
    formatted_code = []
    indent_level = 0
    for line in lines:
        stripped_line = line.strip()
        if stripped_line.endswith('}'):
            indent_level -= 1
        formatted_code.append('\t' * indent_level + stripped_line)
        if stripped_line.endswith('{'):
            indent_level += 1
    
    return '\n'.join(formatted_code)


def printCodePairSideBySide(code1, code2):
    # Split each code snippet by lines
    code1_lines = code1.splitlines()
    code2_lines = code2.splitlines()

    # Set the column width for each snippet
    col_width = 80

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