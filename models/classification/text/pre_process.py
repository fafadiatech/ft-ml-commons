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

def pre_process(directory):
    """
    this method is used to pre-process data, it does

    1. Reads JSON files
    2. Returns Pandas Frame with Text and Label
    """
    results = []
    MAX_WORDS = 0
    for current in os.listdir(directory):
        file_to_read = os.path.join(directory, current)
        data = json.loads(open(file_to_read).read())
        for instance in data:
            row = {}
            row['title'] = instance['title']
            row['blurb'] = instance['blurb']
            row['text'] = cleanup_text(instance['title']) + " " + cleanup_text(instance['blurb'])
            if MAX_WORDS != 0:
                row['text'] = " ".join(row['text'].split()[:min(len(row['text']), MAX_WORDS)])
            else:
                row['text'] = " ".join(row['text'].split())
            row['target'] = file_to_read.split("/")[1].replace(".json", "")
            results.append(row)
    return pd.DataFrame(results)