"""
A class for basic vocab operations.
"""

from __future__ import print_function
import os
import random
import numpy as np
import pickle

from utils import constant

# set random seed
random_seed = 1234
random.seed(random_seed)
np.random.seed(random_seed)

def build_embedding(wv_file, vocab, wv_dim):
    vocab_size = len(vocab)
    emb = np.random.uniform(-1, 1, (vocab_size, wv_dim))
    emb[constant.PAD_ID] = 0 # <PAD> should be all 0

    w2id = {w: i for i, w in enumerate(vocab)}
    with open(wv_file, encoding="utf-8") as f:
        for line in f:
            elems = line.split()
            token = "".join(elems[0: -wv_dim])
            if token in w2id:
                emb[w2id[token]] = [float(v) for v in elems[-wv_dim:]]
    return emb

def load_glove_vocab(file, wv_dim):
    """Load all words from glove
    """
    vocab = set()
    with open(file, encoding='utf-8') as f:
        for line in f:
            elems = line.split()
            token = "".join(elems[0: -wv_dim])
            vocab.add(token)
    return vocab

class Vocab(object):

    def __init__(self, filename, load=False, word_counter=None, threshold=0):
        if load:
            assert os.path.exists(filename), "Vocab file does not exist at " + filename
            # load from file and ignore all other params
            self.id2word, self.word2id = self.load(filename)
            self.size = len(self.word2id)
            print("Vocab size {} loaded from file {}".format(self.size, filename))
        else:
            print("Creating vocab from scratch ...")
            assert word_counter is not None, "word counter is not provided for vocab creation."
            self.word_counter = word_counter
            if threshold > 1:
                # remove words that occur less than threshold
                self.word_counter = dict([(k, v) for k, v in self.word_counter.items() if v >= threshold])
            self.id2word = sorted(word_counter, key=lambda k:self.word_counter[k], reverse=True)
            # add PAD_TOKEN and UNK_TOKEN to id2word
            self.id2word = [constant.PAD_TOKEN, constant.UNK_TOKEN] + self.id2word
            self.word2id = {v: k for k, v in self.id2word.items()}
            self.size = len(self.id2word)
            self.save(filename)
            print("Vocab size {} saved to file {}.".format(self.size, filename))
    
    def load(self, filename):
        with open(filename, 'rb') as fin:
            id2word = pickle.load(fin)
            word2id = dict([(id2word[idx], idx) for idx in range(len(id2word))])
        return id2word, word2id
    
    def save(self, filename):
        if os.path.exists(filename):
            print("Overwriting old vocab file at " + filename)
            os.remove(filename)
        with open(filename, 'wb') as fout:
            pickle.dump(self.id2word, fout)
        return None
    
    def map(self, token_list):
        return [self.word2id[w] if w in self.word2id else constant.UNK_ID for w in token_list]
    
    def unmap(self, idx_list):
        return [id2word[x] for x in idx_list]
    
    def get_embedding(self, word_vector=None, dim=100):
        self.embeddings = 2 * constant.EMB_INIT_RANGE * np.random.rand(self.size, dim) - constant.EMB_INIT_RANGE
        if word_vector is not None:
            assert len(list(word_vector.values())[0]) == dim, \
                "Word vectors does not have requires dimension {}".format(dim)
            for w, idx in self.word2id.items():
                if w in word_vector:
                    self.embeddings[idx] = np.asarray(word_vector[w])
        return self.embeddings