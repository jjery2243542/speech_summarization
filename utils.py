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
    def __init__(self, hdf5_path='/home/jjery2243542/datasets/summary/structured/15673_100_20/cnn_120_15.hdf5'):
        self.datasets = h5py.File(hdf5_path, 'r')

    def make_batch(self, num_datapoints=None, batch_size=32, dataset_type='train'):
        x_path = dataset_type + '/x'
        y_path = dataset_type + '/y'
        if not num_datapoints:
            num_datapoints = self.datasets[x_path].shape[0]
        for i in range(num_datapoints // batch_size + 1):
            l = i * batch_size
            r = min((i + 1) * batch_size, num_datapoints)
            batch_x = self.datasets[x_path][l:r]
            batch_y = self.datasets[y_path][l:r]
            yield batch_x, batch_y

class Vocab(object):
    def __init__(self, vocab_path='/home/jjery2243542/datasets/summary/structured/15673_100_20/vocab.pkl'):
        with open(vocab_path, 'rb') as f_in:
            self.word2idx = pickle.load(f_in)
        self.idx2word = {v:k for k, v in self.word2idx.items()}

    def decode(self, idx_seq, unk_map):
        """
        input: idx sequence
        output: sentenence
        """
        words = []
        # invert unk_map
        unk2word = {v:k for k, v in unk_map.items()}
        for idx in idx_seq:
            word = self.idx2word[idx]
            # check whether it's unk
            if word in unk2word:
                words.append(unk2word[word])
            elif word == '<EOS>':
                break
            elif word not in ['<PAD>', '<BOS>', '<UNK_OTHER>']:
                words.append(word)
        sent = ' '.join(words)
        return sent
            
    def size(self):
        return len(self.word2idx)

if __name__ == '__main__':
    ## test hps
    #hps = Hps()
    #hps.dump('./hps/default.json')

    ## test data_generator
    dg = DataGenerator()
    vocab = Vocab()
    with open('/home/jjery2243542/datasets/summary/structured/15673_100_20/cnn_120_15.hdf5.unk.json', 'r') as f_json:
        all_unk_map = json.load(f_json)
    for i, (batch_x, batch_y) in enumerate(dg.make_batch(batch_size=5, num_datapoints=12, dataset_type='train')):
        print(batch_x.shape, batch_y.shape)
        for j, x in enumerate(batch_x):
            print(vocab.decode(x, all_unk_map['train'][i * 5 + j]))
        for j, y in enumerate(batch_y):
            print(vocab.decode(y, all_unk_map['train'][i * 5 + j]))
        if i > 5:
            break
        
