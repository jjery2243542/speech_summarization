import json
import h5py
import pickle
import os
from collections import defaultdict
from collections import namedtuple
import numpy as np
import nltk
import math

class Hps(object):
    def __init__(self):
        self.hps = namedtuple('hps', [
            'lr', 
            'decay_steps', 
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
        default = [0.15, 10000, 1, 512, 300, 0.8, 32, 80, 15, 7, 1]
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
    def __init__(self, hdf5_path='/home/jjery2243542/datasets/summary/structured/15673_100_20/giga_80_15.hdf5'):
        self.datasets = h5py.File(hdf5_path, 'r')

    def size(self, dataset_type='train'):
        return self.datasets[dataset_type + '/x'].shape[0]

    def make_batch(self, num_datapoints=None, batch_size=32, dataset_type='train'):
        x_path = dataset_type + '/x'
        y_path = dataset_type + '/y'
        if not num_datapoints:
            num_datapoints = self.datasets[x_path].shape[0]
        for i in range(math.ceil(num_datapoints / batch_size)):
            l = i * batch_size
            r = min((i + 1) * batch_size, num_datapoints)
            batch_x = self.datasets[x_path][l:r]
            batch_y = self.datasets[y_path][l:r]
            yield batch_x, batch_y

class Vocab(object):
    def __init__(self, vocab_path='/home/jjery2243542/datasets/summary/structured/15673_100_20/vocab.pkl', unk_map_path='/home/jjery2243542/datasets/summary/structured/15673_100_20/giga_80_15.hdf5.unk.json'):
        with open(vocab_path, 'rb') as f_in:
            self.word2idx = pickle.load(f_in)
        self.idx2word = {v:k for k, v in self.word2idx.items()}
        # <PAD>, <BOS>, <EOS>
        self.num_symbols = 3
        self.num_unks = 20

        # read unk_map json
        with open(unk_map_path, 'r') as f_json:
            self.all_unk_map = json.load(f_json)

    def batch_bleu(self, hypo_sents, ref_sents):
        # turn to bleu format
        hyp = [sent.strip().split() for sent in hypo_sents]
        ref = [[sent.strip().split()] for sent in ref_sents]
        return nltk.translate.bleu_score.corpus_bleu(ref, hyp)
        
    def decode_batch(self, idx_seqs, batch_idx, batch_size, dataset_type='valid'):
        sents = []
        for i, idx_seq in enumerate(idx_seqs):
            sent = self.decode(idx_seq, self.all_unk_map[dataset_type][batch_idx * batch_size + i])
            sents.append(sent)
        return sents
        
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
    for i, (batch_x, batch_y) in enumerate(dg.make_batch(batch_size=5, num_datapoints=12, dataset_type='valid')):
        print(vocab.decode_batch(batch_x, i, 5))
        print(vocab.decode_batch(batch_y, i, 5))
        if i > 5:
            break
        
