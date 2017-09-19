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
from utils import DataGenerator
from utils import Vocab 

if __name__ == '__main__':
    # main function for decode
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '-input_path')
    parser.add_argument('-o', '-output_path')
    parser.add_argument('-dataset_type', default='valid')
    args = parser.parse_args()

    dg = DataGenerator()
    vocab = Vocab()
    vocab.decode_docs(args.i, args.o, args.dataset_type)
