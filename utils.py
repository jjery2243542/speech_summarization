import json
import h5py
import pickle
import os
from collections import defaultdict
import numpy as np

class Hps(object):
    def __init__(self,
                 lr=0.3,
                 hidden_dim=256,
                 embedding_dim=300,
                 keep_prob=0.8,
                 batch_size=32,
                 encoder_length=100,
                 decoder_length=15
                 nll_epochs=7,
                 coverage_epochs=1):
        self.lr = lr
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.keep_prob = keep_prob
        self.batch_size = batch_size
        self.encoder_length = encoder_length
        self.decoder_length = decoder_length
        self.nll_epochs = nll_epochs
        self.coverage_epochs = coverage_epochs

    def load(self, path):
        with open(path, 'r') as f_json:
            hps_json = json.load(f_json)
        self.lr = hps_json['lr']
        self.hidden_dim = hps_json['hidden_dim']
        self.embedding_dim = hps_json['embedding_dim']
        self.keep_prob = hps_json['keep_prob']
        self.batch_size = hps_json['batch_size']
        self.encoder_length = hps_json['encoder_length']
        self.decoder_length = hps_json['decoder_length']
        self.nll_epochs = hps_json['nll_epochs']
        self.coverage_epochs = hps_json['coverage_epochs']

    def dump(self, path):
        hps_json = {
            'lr':self.lr, 
            'hidden_dim':self.hidden_dim,
            'embedding_dim':self.embedding_dim,
            'keep_prob':self.keep_prob,
            'batch_size':self.batch_size,
            'encoder_length':self.encoder_length,
            'decoder_length':self.decoder_length,
        }
        with open(path, 'w') as f_json:
            json.dump(hps_json, f_json, indent=4, separators=(',', ': '))
        
class DataGenerator(object):
    def __init__(self, hdf5_path):
        self.datasets = h5py.File(hdf5_path)

    def make_batch(self, num_datapoints=None, batch_size=32, dataset_type='train'):
        x_path = dataset_type + '/x'
        y_path = dataset_type + '/y'
        if not num_datapoints:
            num_datapoints = self.datasets[x_path].shape[0]
        for i in range(num_datapoints / batch_size + 1):
            l = i * batch_size
            r = (i + 1) * batch_size
            batch_x = self.datasets[x_path][l:r]
            batch_y = self.datasets[y_path][l:r]
            yield batch_x, batch_y

class Vocab(object):
    def __init__(self, vocab_path):
        with open(load_vocab_path, 'rb') as f_in:
            self.word2idx = pickle.load(f_in)
        self.idx2word = {v:k for k, v in self.word2idx.items()}
    def decode(self, idx_seqs):
        """
        input: idx sequences
        output: sentenences
        """
        for seq in idx_seqs:
            
    def size(self):
        return len(self.word2idx)
