import sys
import os
import subprocess
import numpy as np
import glob
import h5py
from pydub import AudioSegment
import pandas as pd
'''
input_dir: xxxxx.mp3
output_path: h5py file for all MFCC feature
'''
def mp3_to_wav(mp3_path, wav_path):
    audio = AudioSegment.from_mp3(mp3_path)
    audio = audio.set_frame_rate(22000)
    audio.export(wav_path, format='wav')

def opensmile_prosodic(tmp_dir, wav_path, feat_path):
    subprocess.run(
        ['SMILExtract',
        '-C',
        '/home_local/opensmile-2.3.0/config/prosodyShs2.conf',
        '-I',
        wav_path, 
        '-O',
        feat_path,
        '-nologfile'
        ]
    )

def csv2npy(feat_path):
    # only last three column are useful
    # sep is ;
    df = pd.read_csv(feat_path, sep=';')
    data = np.array(df)
    data = data[:,2:]
    return data


'''
use openSMILE instead.
'''
def kaldi_mfcc(tmp_dir, wav_path, feat_path):
    # write wav path to rspecifier
    tmp_scp_path = os.path.join(tmp_dir, 'tmp.scp')
    with open(tmp_scp_path, 'w') as f_out:
        f_out.write('{} {}\n'.format(0, wav_path))
    feat13_path = os.path.join(tmp_dir, '13.ark')
    subprocess.run(
        ['compute-mfcc-feats', 
        '--use-energy=false', 
        '--channel=0', 
        '--sample-frequency=22000',
        'scp:{}'.format(tmp_scp_path),
        'ark,t:{}'.format(feat13_path),
        ]
    )
    feat39_path = os.path.join(tmp_dir, '39.ark')
    subprocess.run(
        ['add-deltas', 
        'ark,t:{}'.format(feat13_path),
        'ark,t:{}'.format(feat39_path)
        ]
    )
    cmvn_path = os.path.join(tmp_dir, 'cmvn.ark')
    subprocess.run(
        ['compute-cmvn-stats', 
        'ark:{}'.format(feat39_path),
        'ark,t:{}'.format(cmvn_path),
        ]
    )
    tmp_feat_path = os.path.join(tmp_dir, feat_path)
    subprocess.run(
        ['apply-cmvn',
        '--norm-vars=true',
        'ark:{}'.format(cmvn_path),
        'ark,t:{}'.format(feat39_path),
        'ark,t:{}'.format(tmp_feat_path),
        ]
    )
    
    
if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('feat_extraction.py [input_dir] [output_path] [feature_type=prosody|emotion]')
        exit(0)
    input_dir = sys.argv[1]
    output_path = sys.argv[2]

    # mkdir tmp_dir
    tmp_dir = '/tmp/feature_tmp/'
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)

    with h5py.File(output_path, 'w') as f_h5:
        for file_idx, filename in enumerate(sorted(glob.glob(os.path.join(input_dir, '*.mp3')))):
            print('process {} files'.format(file_idx))
            # index is the mp3 file number
            index = filename.split('/')[-1]
            index = index[:-4]
            # create hdf5 group
            # convert to wav in tmp_dir
            wav_path = os.path.join(tmp_dir, 'tmp.wav')
            mp3_to_wav(filename, wav_path)
            feat_path = os.path.join(tmp_dir, 'feature.csv')
            opensmile_prosodic(tmp_dir, wav_path, feat_path)
            data = csv2npy(feat_path)
            f_h5.create_dataset(index, data=data, dtype=np.float32)
