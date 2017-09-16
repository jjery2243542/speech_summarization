import sys
import os
import subprocess
import numpy as np
import glob
import h5py

'''
input_dir: xxxxx.mp3
output_path: h5py file for all MFCC feature
'''

if __name__ == '__main__':
    tmp_dir = '/tmp/kaldi_tmp/'
    if len(sys.argv) < 3:
        print('feat_extraction.py [input_dir] [output_path]')
    for filename in 

