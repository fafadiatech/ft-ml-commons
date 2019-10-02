# this file contains code for generating features

import re
import numpy as np
import pandas as pd

# dependencies for word2vec
from gensim.models import KeyedVectors
from gensim.test.utils import datapath

def gen_counting_features(dataset):
    """
    this is used to generate various counting features
    """
    results = []
    for _, row in dataset.iterrows():
        rec = {}
        rec['title_char_len'] = len(row['title'])
        rec['title_word_len'] = len(row['title'].split())
        rec['title_density'] = rec['title_char_len'] / float(rec['title_word_len'])
        rec['blurb_char_len'] = len(row['blurb'])
        rec['blurb_word_len'] = len(row['blurb'].split())
        rec['blurb_density'] = rec['blurb_char_len'] / float(rec['blurb_word_len'])
        results.append(rec)
    return pd.DataFrame(results)

def word2_vec_embeddings(doc, model_path, DIM=300):
    """
    this function is used to get additive embedding
    on all words in document

    here `model_path` is binary file for word2vec's 
    pre-trained model {e.g. GoogleNews-vectors-negative300.bin}

    """
    stop_word_filter = lambda x: x not in stoplist
    word2vec = KeyedVectors.load_word2vec_format(datapath(model_path), binary=True)
    vector = [np.zeros(DIM)]

    for current in list(set(filter(stop_word_filter, doc.split()))):
        if current in word2vec:
            vector = np.add(vector , word2vec[current])
        else:
            vector = np.add(vector, [np.zeros(DIM)])
    return vector

def gen_embedding_features(dataset):
    results = []
    for _, row in dataset.iterrows():
        vector = word2_vec_embeddings(row['text']).flatten()
        results.append(vector)
    return results