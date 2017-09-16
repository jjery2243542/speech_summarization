import os
import random
import sys
import numpy as np

if __name__ == '__main__':
    # $1: input directory with content and title text file, index.txt
    # $2: output directory with content/[train.txt, valid.txt, test.txt] title/[train.txt, valid.txt, test.txt] index/[train.txt, valid.txt, test.txt]
    # $3: train, valid, test fraction(ex. 0.8,0.1,0.1)
    if len(sys.argv) < 4:
        print('usage: python3 split.py [input_dir] [output_dir] [fractions, ex.0.8,0.1,0.1]')
        exit(0)
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    fracs = [float(percentage) for percentage in sys.argv[3].strip().split(',', 3)]
    with open(os.path.join(input_dir, 'content.txt'), 'r') as f_content, open(os.path.join(input_dir, 'title.txt'), 'r') as f_title, open(os.path.join(input_dir, 'index.txt'), 'r') as f_index:
        # read files
        contents = [line.strip() for line in f_content] 
        titles = [line.strip() for line in f_title]
        indexs = [line.strip() for line in f_index]

    # for shuffle
    randomize = np.arange(len(contents))
    np.random.shuffle(randomize)
    divide_point = [int(len(randomize) * fracs[0]), int(len(randomize) * (fracs[0] + fracs[1]))]
    # mkdir if not exist
    content_dir = os.path.join(output_dir, 'content')
    title_dir = os.path.join(output_dir, 'title')
    index_dir = os.path.join(output_dir, 'index')
    if not os.path.exists(content_dir):
        os.mkdir(content_dir)
    if not os.path.exists(title_dir):
        os.mkdir(title_dir)
    if not os.path.exists(index_dir):
        os.mkdir(index_dir)
    # write files 
    with open(os.path.join(content_dir, 'train.txt'), 'w') as f_content, open(os.path.join(title_dir, 'train.txt'), 'w') as f_title, open(os.path.join(index_dir, 'train.txt'), 'w') as f_index:
        for data_idx in randomize[:divide_point[0]]:
            f_content.write(contents[data_idx] + '\n')
            f_title.write(titles[data_idx] + '\n')
            f_index.write(indexs[data_idx] + '\n')

    with open(os.path.join(content_dir, 'valid.txt'), 'w') as f_content, open(os.path.join(title_dir, 'valid.txt'), 'w') as f_title, open(os.path.join(index_dir, 'valid.txt'), 'w') as f_index:
        for data_idx in randomize[divide_point[0]:divide_point[1]]:
            f_content.write(contents[data_idx] + '\n')
            f_title.write(titles[data_idx] + '\n')
            f_index.write(indexs[data_idx] + '\n')

    with open(os.path.join(content_dir, 'test.txt'), 'w') as f_content, open(os.path.join(title_dir, 'test.txt'), 'w') as f_title, open(os.path.join(index_dir, 'test.txt'), 'w') as f_index:
        for data_idx in randomize[divide_point[1]:]:
            f_content.write(contents[data_idx] + '\n')
            f_title.write(titles[data_idx] + '\n')
            f_index.write(indexs[data_idx] + '\n')
