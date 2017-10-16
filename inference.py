from utils import DataGenerator
from utils import Hps
from utils import Vocab
from pointer_model import PointerModel
import argparse

def predict(model, iterator, output_path='result_index.txt'):
    print('starting inference...')
    with open(output_path, 'w') as f_out:
        for i, (batch_x, batch_y) in enumerate(iterator):
            all_result = model.predict_step(batch_x)
            for result in all_result:
                for word_idx in result:
                    f_out.write('{} '.format(word_idx))
                f_out.write('\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-hps_path', default='./hps/cd_v3.json')
    parser.add_argument('-vocab_path', default='/home/jjery2243542/datasets/summary/structured/26693_50_30/vocab.pkl')
    parser.add_argument('-model_path', default='./model/model.ckpt-2999')
    parser.add_argument('-dataset_path', default='/home/jjery2243542/datasets/summary/structured/26693_50_30/giga_40_10.h5')
    parser.add_argument('-dataset_type', default='valid')
    parser.add_argument('-output_path', default='result.txt')
    args = parser.parse_args()
    hps = Hps()
    hps.load(args.hps_path)
    hps_tuple = hps.get_tuple()
    print(hps_tuple)
    vocab = Vocab(args.vocab_path, args.dataset_path + '.unk.json')
    data_generator = DataGenerator(args.dataset_path)
    model = PointerModel(hps_tuple, vocab)
    model.load_model(args.model_path)
    dg = DataGenerator(args.dataset_path)
    iterator = dg.iterator(
        batch_size=hps_tuple.batch_size, 
        dataset_type=args.dataset_type, 
        infinite=False, 
        shuffle=False
    )
    predict(model, iterator, args.output_path)
