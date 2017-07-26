import numpy as np
import hdf5
import pickle
import os
import string

class Preprocessor(object):
    def __init__(self):
        
    def check_punc(self, s):
        """
        check whether the whole word is punctuation
        """
        s = s.strip()
        is_punc = [1 if c in string.punctuation else 0 for c in s]
        if sum(is_punc) == len(s):
            return False
        else:
            return True
            
    def make_datasets(self, text_root_dir, dump_path, unk_map_path, content_length=120, title_length=20):
         '''
         root_dir-content-train.txt
                         -valid.txt
                         -test.txt
                 -title-train.txt
                       -valid.txt
                       -test.txt
         '''
         unk_map = {'train':[], 'valid':[], 'test':[]}
         with h5py.File(dump_path, 'w') as f_hdf5:
             for dataset in ['train', 'valid', 'test']:
                grp = f_hdf5.create_group(dataset)
                with open(os.path.join(root_dir, 'content/' + dataset + '.txt')) as f_content, open(os.path.join(root_dir, 'title/' + dataset + '.txt')) as f_title:
                    X = []
                    Y = []
                    for idx, (content, title) in enumerate(zip(f_content, f_title)):
                        words = [word for word in content.strip().split() if self.check_punc(word)][:content_length]
                        x = []
                        # mapping the unk to words
                        unk = {}
                        for word in words:
                            if word in self.word2idx:
                               x.append(self.word2idx[word]) 
                            else:
                                unk_idx = len(unk)
                                if 
                                
    def dump_vocab(self, path):
        """
        dump vocab to disk
        """
        with open(path, 'wb') as f_out:
            pickle.dump(self.word2idx, save_vocab_path)
        print('dump vocab file to {}'.format(path))
    
    def load_vocab(self, path):
        with open(path, 'rb') as f_in:
            self.word2idx = pickle.load(f_in)
        print('load vocab file from {}'.format(path))

    def get_vocab(self, root_dir, min_occur=2000, num_unk=20):
         '''
         root_dir-content-train.txt
                         -valid.txt
                         -test.txt
                 -title-train.txt
                       -valid.txt
                       -test.txt
         '''
        self.word2idx = {'<pad>':0, '<bos>':1, '<eos>':2}
        # add unk to vocab
        for i in range(num_unks):
            self.word2idx['<unk' + '_{}>'.format(i)] = len(self.word2idx)
        self.word2idx['<unk_other>'] = len(self.word2idx)
         count_dict = defaultdict(lambda: 0)
         for dataset in ['train', 'valid', 'test']:
            with open(os.path.join(root_dir, 'content/' + dataset + '.txt')) as f_content, open(os.path.join(root_dir, 'title/' + dataset + '.txt')) as f_title:
                for content, title in zip(f_content, f_title):
                    for word in content.strip().split():
                        count_dict[word] += 1
            for word in count_dict:
                if count_dict[word] >= min_occur:
                    self.word2idx[word] = len(self.word2idx)
         print('vocab_size={}'.format(len(self.word2idx)))
         return 
