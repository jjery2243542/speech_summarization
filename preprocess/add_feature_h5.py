import h5py
import sys
import os
import numpy as np

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('usage: python3 add_feature_h5.py [length] [h5py_file] [feature_h5_file] [index_dir]')
        exit(0)
    length = int(sys.argv[1])
    h5_path = sys.argv[2]
    feature_path = sys.argv[3]
    index_dir = sys.argv[4]
    index_paths = [os.path.join(index_dir, filename) for filename in ['train.txt', 'valid.txt', 'test.txt']]

    with h5py.File(h5_path, 'r+') as f_h5, h5py.File(feature_path, 'r') as f_feat:
        for index_path, dataset in zip(index_paths, ['train', 'valid', 'test']):
            print('processing {}'.format(dataset))
            with open(index_path, 'r') as f_index:
                s = []
                for idx, line in enumerate(f_index):
                    index = line.strip()
                    feature = np.zeros([length, 3], dtype=np.float32)
                    feature_length = length if f_feat[index].shape[0] > length else f_feat[index].shape[0]
                    feature[:feature_length] += f_feat[index][:feature_length]
                    s.append(feature)
                    if idx % 100 == 0:
                        print('process {} datapoints'.format(idx))
                s = np.array(s, dtype=np.float32)
                f_h5['{}/s'.format(dataset)] = s

             


