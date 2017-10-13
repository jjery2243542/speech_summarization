from utils import DataGenerator
import time
import datetime
import argparse
from pointer_model import PointerModel
from utils import Vocab
from utils import Hps

def train_loop(
        model, 
        data_generator, 
        iterations, 
        log_fp,
        model_path, 
        batch_size=16, 
        coverage=False, 
        patience=5, 
        min_delta=0.01):
    train_loss = 0.
    # for early stopping
    prev_val_loss = 1e7
    iteration = 0
    start_time = time.time()
    # make iterator
    train_iterator = data_generator.iterator(
        batch_size=batch_size, 
        dataset_type='train',
        infinite=True, 
        shuffle=True,
    )
    for iteration, (batch_x, batch_y) in enumerate(train_iterator):
        loss = model.train_step(batch_x, batch_y, coverage=coverage)
        train_loss += loss
        avg_train_loss = train_loss / (iteration + 1)
        end_time = time.time()
        time_span = int(end_time - start_time)
        formatted_time = datetime.timedelta(seconds=time_span)
        # for print infomation
        slot_value = (
            iteration,
            iterations,
            coverage,
            loss,
            avg_train_loss,
            formatted_time,
        )
        print('step [%06d/%06d], coverage=%r, loss: %.4f, avg_loss: %.4f, time: %s\r' % slot_value, end='')
        # valid
        if iteration % 3000 == 0 and iteration != 0:
            valid_iterator = data_generator.iterator(
                num_batchs=200,
                batch_size=batch_size,
                dataset_type='valid',
                infinite=False,
                shuffle=False,
            )
            val_loss = valid(model, valid_iterator)
            # write to log
            log_fp.write('%06d,%r,%.4f,%.4f\n' % (iteration, coverage, avg_train_loss, val_loss))
            log_fp.flush()
            # save model
            model.save_model(model_path, global_step=iteration)
            if (val_loss - prev_val_loss) > min_delta:
                patience -= 1
        # finished or early stop
        if iteration + 1 >= iterations or patience == 0:
            model.save_model(model_path, global_step=iteration)
            break
             
def train(model, data_generator, log_file_path, model_path):
    print('start training...')
    with open(log_file_path, 'w') as log_fp:
        log_fp.write('iteration,coverage,train_loss,val_loss\n')
        log_fp.flush()
        train_loop(
            model, 
            data_generator, 
            model._hps.nll_iterations, 
            log_fp, 
            model_path, 
            model._hps.batch_size, 
            coverage=False, 
            patience=10, 
            min_delta=0.01
        )
        train_loop(
            model, 
            data_generator, 
            model._hps.coverage_iterations, 
            log_fp, 
            model_path, 
            model._hps.batch_size, 
            coverage=True, 
            patience=10, 
            min_delta=0.01
        )

def valid(model, iterator):
    total_loss = 0. 
    step = 0
    for batch_x, batch_y in iterator:
        loss = model.valid_step(batch_x, batch_y)
        total_loss += loss
        step += 1
    avg_loss = total_loss / step
    return avg_loss 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-hps_path', default='./hps/cd.json')
    parser.add_argument('-dataset_path', default='/home/jjery2243542/datasets/summary/structured/26693_50_30/cd_400_100.h5')
    parser.add_argument('--pretrain_wordvec', action='store_true')
    parser.add_argument('-npy_path', default='/home/jjery2243542/datasets/summary/structured/26693_50_30/glove.npy')
    parser.add_argument('-log_file_path', default='./log.txt')
    parser.add_argument('-write_model_path', default='./model/model.ckpt')
    parser.add_argument('--load_model')
    parser.add_argument('-read_model_path', default='./model/model.ckpt')
    parser.add_argument('-vocab_path', default='/home/jjery2243542/datasets/summary/structured/26693_50_30/vocab.pkl')
    args = parser.parse_args()
    # get hps
    hps = Hps()
    hps.load(args.hps_path)
    hps_tuple = hps.get_tuple()
    print(hps_tuple)
    vocab = Vocab(args.vocab_path, args.dataset_path + '.unk.json')
    data_generator = DataGenerator(args.dataset_path)
    model = PointerModel(hps_tuple, vocab)
    if args.pretrain_wordvec:
        model.init(npy_path=args.npy_path)
    else:
        model.init()
    if args.load_model:
        model.load_model(args.read_load_model)
    train(
        model=model,
        data_generator=data_generator,
        log_file_path=args.log_file_path, 
        model_path=args.write_model_path,
    )
