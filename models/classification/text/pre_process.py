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