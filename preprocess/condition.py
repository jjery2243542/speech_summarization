import sys
import os
if __name__ == '__main__':
# pick the data satisfied the conditions
# $1: input directory with content.txt, title.txt, index.txt
# $2: output directory with content.txt, title.txt, index.txt
    if len(sys.argv) < 3:
        print('usage: python3 condition.py [input_dir] [output_dir]')
        exit(0)
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    min_intersection = 2
    min_content_words = 30
    min_title_words = 5

    contents = []
    titles = []
    indexs = []
    with open(os.path.join(input_dir, 'content.txt')) as f_content, open(os.path.join(input_dir, 'title.txt')) as f_title, open(os.path.join(input_dir, 'index.txt')) as f_index:
        for index, content, title in zip(f_index, f_content, f_title):
            keep = True
            content_words = [word for word in content.strip().split()]
            title_words = [word for word in title.strip().split()]
            # check intersection
            if len(set(content_words) & set(title_words)) < min_intersection:
                keep = False
            # check word count
            if len(content_words) < min_content_words or len(title_words) < min_title_words:
                keep = False
            if keep:
                contents.append(content.strip())
                titles.append(title.strip())
                indexs.append(index.strip())

    print('{} documents remains'.format(len(contents)))

    # write to files
    with open(os.path.join(output_dir, 'content.txt'), 'w') as f_content, open(os.path.join(output_dir, 'title.txt'), 'w') as f_title, open(os.path.join(output_dir, 'index.txt'), 'w') as f_index:
        f_content.write('\n'.join(contents))
        f_title.write('\n'.join(titles))
        f_index.write('\n'.join(indexs))

