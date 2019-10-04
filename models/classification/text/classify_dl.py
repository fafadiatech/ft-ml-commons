import os
import argparse
import numpy as np


from pre_process import load_dataset

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
from keras.initializers import Constant

embeddings_index = {}

GLOVE_DIR = "/home/tensor/Code/machine_learning/pre_trained_models/glove"
MAX_SEQUENCE_LENGTH = 1000
MAX_NUM_WORDS = 20000
EMBEDDING_DIM = 100

parser = argparse.ArgumentParser()
parser.add_argument("train", help="training dataset in TSV format {label, text}")
parser.add_argument("test", help="training dataset in TSV format {label, text}")
args = parser.parse_args()

print("Loading GloVe vectors")
with open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt')) as f:
    for line in f:
        word, coefs = line.split(maxsplit=1)
        coefs = np.fromstring(coefs, 'f', sep=' ')
        embeddings_index[word] = coefs
print("Completed loading")

def gen_sequence(texts):
    """
    this method is used to generate sequences
    from embedding vectors
    """
    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))
    
    return pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH), word_index

if __name__ == "__main__":
    X_train, y_train = load_dataset(args.train)
    X_test, y_test = load_dataset(args.test)

    print("Vectorizing...")
    X_train_vectorized, word_index = gen_sequence(X_train)
    y_train_vectorized = to_categorical(np.asarray(y_train))
    print("Creating embedding matrix")
    
    num_words = min(MAX_NUM_WORDS, len(word_index) + 1)
    embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
    for word, i in word_index.items():
        if i >= MAX_NUM_WORDS:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    print("Vectorizing test...")
    X_test_vectorized, word_index = gen_sequence(X_test)
    y_test_vectorized = to_categorical(np.asarray(y_test))

    print("Compiling model")
    embedding_layer = Embedding(num_words, EMBEDDING_DIM, embeddings_initializer=Constant(embedding_matrix), input_length=MAX_SEQUENCE_LENGTH, trainable=False)
    
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    x = Conv1D(128, 5, activation='relu')(embedded_sequences)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    x = GlobalMaxPooling1D()(x)
    x = Dense(128, activation='relu')(x)
    preds = Dense(21, activation='softmax')(x)

    model = Model(sequence_input, preds)
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])
    model.fit(X_train_vectorized, y_train_vectorized, batch_size=128, epochs=10, validation_data=(X_train_vectorized, y_train_vectorized))
