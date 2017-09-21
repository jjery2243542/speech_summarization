import json
import h5py
import pickle
import os
from collections import defaultdict
from collections import namedtuple
import numpy as np
import nltk
import math
import argparse

class Hps(object):
    def __init__(self):
        self.hps = namedtuple('hps', [
            'lr',
            'lamb',
            'max_grad_norm',
            'hidden_dim', 
            'embedding_dim', 
            'keep_prob', 
            'batch_size', 
            'encoder_length', 
            'decoder_length',
            'nll_epochs',
            'coverage_epochs']
        )
        default = [0.15, 0.8, 2, 256, 300, 0.8, 32, 80, 15, 10, 2]
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
    def __init__(self, 
    vocab_path='/home/jjery2243542/datasets/summary/structured/26693_50_30/vocab.pkl', 
    unk_map_path='/home/jjery2243542/datasets/summary/structured/26693_50_30/cd_400_100.h5.unk.json'):
        with open(vocab_path, 'rb') as f_in:
            self.word2idx = pickle.load(f_in)
        self.idx2word = {v:k for k, v in self.word2idx.items()}
        # <PAD>, <BOS>, <EOS>
        self.num_symbols = 3
        self.num_unks = 30

        # read unk_map json
        with open(unk_map_path, 'r') as f_json:
            self.all_unk_map = json.load(f_json)

    def batch_bleu(self, hypo_sents, ref_sents):
        # turn to bleu format
        hyp = [sent.strip().split() for sent in hypo_sents]
        ref = [[sent.strip().split()] for sent in ref_sents]
        return nltk.translate.bleu_score.corpus_bleu(ref, hyp)

    def decode_docs(self, hypo_index_path, hypo_path, dataset_type='valid'):
        with open(hypo_index_path) as f_in, open(hypo_path, 'w') as f_out:
            for line, unk_map in zip(f_in, self.all_unk_map[dataset_type]):
                idx_seq = [int(idx) for idx in line.strip().split()]
                sent = self.decode(idx_seq, unk_map)
                f_out.write(sent + '\n')
        
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
    hps = Hps()
    hps.dump('./hps/default.json')
    # main function for decode
    #parser = argparse.ArgumentParser()
    #parser.add_argument('-i', '-input_path')
    #parser.add_argument('-o', '-output_path')
    #parser.add_argument('-dataset_type', default='valid')
    #args = parser.parse_args()

    #dg = DataGenerator()
    #vocab = Vocab()
    #vocab.decode_docs(args.i, args.o, args.dataset_type)
