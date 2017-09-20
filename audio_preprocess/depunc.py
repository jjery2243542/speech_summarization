import sys
import string

if __name__ == '__main__':
    # remove punctuation in title
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    with open(input_path) as f_in, open(output_path, 'w') as f_out:
        for line in f_in:
            words = []
            for word in line.strip().split():
                num_punc = sum([1 if c in string.punctuation else 0 for c in word])
                if num_punc < len(word):
                    words.append(word)
            f_out.write('{}\n'.format(' '.join(words)))
