from pyrouge import Rouge155
import subprocess
import sys
import shutil
import os
def split_by_line(file_path, directory):
    with open(file_path) as f_in:
        for i, line in enumerate(f_in):
            with open(os.path.join(directory, 'doc.%07d.txt' % (i)), 'w') as f_out:
                f_out.write(line)

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('usage: python3 eval_rouge.py [ref_file] [hypo_file]')
    ref_path=sys.argv[1]
    hypo_path=sys.argv[2]
    tmp_dir='/tmp/rouge_tmp/'
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.mkdir(tmp_dir)
    model_dir=os.path.join(tmp_dir, 'model')
    sys_dir=os.path.join(tmp_dir, 'system')
    os.mkdir(model_dir)
    os.mkdir(sys_dir)

    # split by line
    split_by_line(ref_path, model_dir)
    split_by_line(hypo_path, sys_dir)

    # rouge
    r = Rouge155()
    r.system_dir = sys_dir
    r.model_dir = model_dir
    r.system_filename_pattern = 'doc.(\d+).txt'
    r.model_filename_pattern = 'doc.#ID#.txt'
    output = r.convert_and_evaluate()
    print(output)
    output_dict = r.output_to_dict(output)
