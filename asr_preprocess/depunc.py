import sys
import string

if __name__ == '__main__':
    # remove punctuation in title
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    with open(input_path) as f_in, open(output_path, 'w') as f_out:
        for line in f_in:
            words = [word for word in line.strip().split() if word not in string.punctuation]
            f_out.write(' '.join(words) + '\n')


