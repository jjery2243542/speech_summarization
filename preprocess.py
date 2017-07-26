import numpy as np
import h5py
import pickle
import os
import string
import argparse

class Preprocessor(object):
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

    def count(self, root_dir, content_threshold=80, title_threshold=15):
        """
        count average word length
        """
        num_data = 0.
        content_word_cnt = 0.
        title_word_cnt = 0.
        content_cnt = 0.
        title_cnt = 0.
        for dataset in ['train', 'valid', 'test']:
            with open(os.path.join(root_dir, 'content/' + dataset + '.txt')) as f_content, open(os.path.join(root_dir, 'title/' + dataset + '.txt')) as f_title:
                for content, title in zip(f_content, f_title):
                    content_words = [word for word in content.strip().split()] 
                    content_word_cnt += len(content_words)
                    if len(content_words) < content_threshold:
                        content_cnt += 1
                    title_words = [word for word in title.strip().split()] 
                    title_word_cnt += len(title_words)
                    if len(title_words) < title_threshold:
                        title_cnt += 1
                    num_data += 1
        print('avg_content_length={}, avg_title_length={}'.format(content_word_cnt / num_data, title_word_cnt / num_data))
        print('content_length < {} ={}, title_length < {} ={}'.format(content_threshold, content_cnt / num_data, title_threshold, title_cnt / num_data))
        return  

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--count', action='store_true')
    parser.add_argument('-doc_path', default='/nfs/Athena/roylu/raw_data/EnglishGigaWord/')
    args = parser.parse_args()
    preprocessor = Preprocessor()
    if args.count:
        preprocessor.count(args.doc_path)
   
