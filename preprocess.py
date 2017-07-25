import numpy as np
import hdf5
import pickle
import os
import string

class Preprocessor(object):
    def __init__(self, min_occur=2000, text_root_dir=None, load_vocab_path=None, save_vocab_path=None):
        if not load_vocab_path:
            self.word2idx = {'<PAD>':0, '<BOS>':1, '<EOS>':2}
            if text_root_dir:
                self._get_vocab(text_root_dir, min_occur)
                if save_vocab_path:
                    with open(save_vocab_path, 'wb') as f_out:
                    pickle.dump(self.word2idx, save_vocab_path)
            else:
                print('should provide either text_path or vocab_path')
                exit(0)
        else:
            with open(load_vocab_path, 'rb') as f_in:
                self.word2idx = pickle.load(f_in)

    def check_punc(self, s):
        s = s.strip()
        is_punc = [1 if c in string.punctuation else 0 for c in s]
        if sum(is_punc) == len(s):
            return False
        else:
            return True
            
    def make_datasets(self, text_root_dir, dump_path):
         '''
         root_dir-content-train.txt
                         -valid.txt
                         -test.txt
                 -title-train.txt
                       -valid.txt
                       -test.txt
         '''
         with h5py.File(dump_path, 'w') as f_hdf5:
             for dataset in ['train', 'valid', 'test']:
                grp = f_hdf5.create_group(dataset)
                with open(os.path.join(root_dir, 'content/' + dataset + '.txt')) as f_content, open(os.path.join(root_dir, 'title/' + dataset + '.txt')) as f_title:
                    x = []
                    y = []
                    for content, title in zip(f_content, f_title):
                        words = [word for word in content.strip().split() if ]
                            
                        
         
        
    def _get_vocab(self, root_dir, min_occur=2000):
         '''
         root_dir-content-train.txt
                         -valid.txt
                         -test.txt
                 -title-train.txt
                       -valid.txt
                       -test.txt
         '''
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
    
