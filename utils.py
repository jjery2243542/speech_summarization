import json
import h5py
import pickle
import os
from collections import defaultdict
from collections import namedtuple
import numpy as np

class Hps(object):
    def __init__(self):
        self.hps = namedtuple('hps', [
            'lr', 
            'decay_step', 
            'decay_rate', 
            'hidden_dim', 
            'embedding_dim', 
            'keep_prob', 
            'batch_size', 
            'encoder_length', 
            'decoder_length',
            'nll_epochs',
            'coverage_epochs']
        )
        default = [0.3, 10000, 0.95, 256, 300, 0.8, 32, 100, 15, 7, 1]
        self._hps = self.hps._make(default)

    def get_tuple(self):
        return self._hps

    def load(self, path):
        with open(path, 'r') as f_json:
            hps_dict = json.load(f_json)
        self._hps = self.hps(**hps_dict)

    def dump(self, path):
        with open(path, 'w') as f_json:
            json.dump(self._hps._asdict(), f_json, indent=4, separators=(',', ': '))
        
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
            print(seq)
            
            
    def size(self):
        return len(self.word2idx)

if __name__ == '__main__':
    hps = Hps()
    print(hps.get_tuple())
    hps.load('test.json')
    print(hps.get_tuple())
