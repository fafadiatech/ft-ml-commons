# this file contains functions to pre-process text

import os
import re
import json
import pandas as pd

def cleanup_text(snippet):
    """
    this method is used to cleanup text
    """
    snippet = snippet.lower().strip()
    snippet = re.sub(r'\n', ' ', snippet)
    snippet = re.sub(r'[^\w\s]+', '', snippet)
    snippet = snippet.replace("\n", "")
    snippet = " ".join(snippet.split())
    return snippet

def load_dataset(file_to_read):
    """
    this method is used to load dataset into pandas frame
    return (train, label)
    """
    df = pd.read_csv(file_to_read, sep="\t", error_bad_lines=False)
    return df.iloc[:, 1], df.iloc[:, 0]