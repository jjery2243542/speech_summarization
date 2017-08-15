import os
import sys
import glob
import xml.etree.ElementTree as ET
def parse_file(filename):
    tree = ET.parse(filename)
    root = tree.getroot()
    sent = ''
    for word in root.iter('word'):
        word = word.text.lower()
        sent += word + ' '
    return sent
    
if __name__ == '__main__':
    # $1: input_dir
    # $2: output_path
    if len(sys.argv) < 3:
        print('usage: python3 merge_xml.py [input_dir] [output_path]')
        exit(0)
    input_dir = sys.argv[1]
    output_path = sys.argv[2]
    with open(output_path, 'w') as f_out:
        for filename in sorted(glob.glob(os.path.join(input_dir, '*.xml'))):
            f_out.write(parse_file(filename) + '\n')


