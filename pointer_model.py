import tensorflow as tf
import numpy as np
from utils import Hps
from utils import Vocab
from utils import DataGenerator
import time
import argparse

class PointerModel(object):
    def __init__(self, hps, vocab):
        self._hps = hps
        self._vocab = vocab
        self.build_graph()
        self.sess = tf.Session()
        self.saver = tf.train.Saver(max_to_keep=2)
        
    def PointerDecoder(self, decoder_inputs, initial_state, encoder_states, cell, feed_previous=False):
        """
        encoder_states shape = [batch_size, length, hidden_dim]
        """
        hidden_dim = self._hps.hidden_dim
        vocab_size = self._vocab.size()
        encoder_inputs = self.x
        embedding_matrix = self.embedding_matrix

        with tf.variable_scope('pointer_decoder') as scope:
            V = tf.get_variable(name='V', shape=[hidden_dim, 1])
            W_h = tf.get_variable(name='W_h', shape=[2 * hidden_dim, hidden_dim])
            W_s = tf.get_variable(name='W_s', shape=[hidden_dim, hidden_dim])
            W_c = tf.get_variable(name='W_c', shape=[hidden_dim])
            b_attn = tf.get_variable(name='b_attn', shape=[hidden_dim])
            # input one-hot for attention
            input_one_hot = tf.one_hot(encoder_inputs, vocab_size)
            encoder_length = encoder_states.get_shape()[1].value
            # flatten to matmul
            flatten_enc = tf.reshape(encoder_states, [-1, 2 * hidden_dim])
            flatten_enc = tf.matmul(flatten_enc, W_h)
            # reshape back to original
            enc = tf.reshape(flatten_enc, [-1, encoder_length, hidden_dim])

            def input_projection(inp, last_attn_context):
                inp = tf.concat([inp, last_attn_context], axis=1)
                return tf.layers.dense(inp, hidden_dim, name='input_projection')

            def gen_layer(inp, state, attn_context):
                inp = tf.concat([inp, state.h, attn_context], axis=1)
                return tf.layers.dense(inp, 1, activation=tf.nn.sigmoid, name='gen_layer')

            def attention(dec, c_t):
                """
                c_t refer to coervage vector
                c_t shape = [batch_size, encoder_length]
                """
                dec = tf.matmul(dec, W_s)
                dec = tf.expand_dims(dec, axis=1)
                cover = tf.expand_dims(c_t, axis=2)
                cover = cover * W_c
                H = tf.nn.tanh(dec + enc + cover + b_attn)
                flatten_H = tf.reshape(H, [-1, hidden_dim])
                attn_weights = tf.matmul(flatten_H, V)
                attn_weights = tf.reshape(attn_weights, [-1, encoder_length])
                attn_weights = tf.nn.softmax(attn_weights)
                expand_attn_weights = tf.expand_dims(attn_weights, axis=1)
                attn_context = tf.matmul(expand_attn_weights, encoder_states)
                attn_context = tf.squeeze(attn_context, axis=1)
                return attn_weights, attn_context

            def get_pointer_distr(attn_weights):
                expand_attn_weights = tf.expand_dims(attn_weights, axis=1)
                expand_pointer_distr = tf.matmul(expand_attn_weights, input_one_hot)
                pointer_distr = tf.squeeze(expand_pointer_distr, axis=1)
                return pointer_distr

            def get_vocab_distr(cell_output, attn_context):
                output_layer_input = tf.concat([cell_output, attn_context], axis=1)
                output_layer_hidden = tf.layers.dense(output_layer_input, hidden_dim, activation=tf.nn.relu, name='output_projection')
                vocab_distr = tf.layers.dense(output_layer_hidden, vocab_size, activation=tf.nn.softmax, name='output_layer')
                return vocab_distr
                
            decoder_outputs = []
            coverage_loss_list = []
            state = initial_state
            attn_coverage = tf.zeros([self.batch_size_tensor, encoder_states.get_shape()[1].value], dtype=tf.float32)
            # do attention for the first time
            attn_weights, attn_context = attention(state.h, attn_coverage)
            for i in range(len(decoder_inputs)):
                if i > 0:
                    scope.reuse_variables()
                elif i > 0 and feed_previous:
                    last_output_id = tf.argmax(decoder_outputs[-1], axis=-1)
                    inp = tf.nn.embedding_lookup(self.embedding_matrix, last_output_id)
                else:
                    last_output_id = decoder_inputs[i]
                    inp = tf.nn.embedding_lookup(self.embedding_matrix, last_output_id)
                cell_input = input_projection(inp, attn_context)
                cell_output, state = cell(cell_input, state)
                attn_weights, attn_context = attention(state.h, attn_coverage)
                coverage_loss_list.append(tf.minimum(attn_coverage, attn_weights))
                attn_coverage += attn_weights

                p_gen = gen_layer(inp, state, attn_context)
                #output_t = get_vocab_distr(cell_output, attn_context)
                output_t = p_gen * get_vocab_distr(cell_output, attn_context) + (1 - p_gen) * get_pointer_distr(attn_weights)
                decoder_outputs.append(output_t)
            coverage_loss = tf.stack(coverage_loss_list, axis=1)
            return decoder_outputs, coverage_loss

    def _add_placeholder(self):
        hps = self._hps
        self.x = tf.placeholder(tf.int32, [None, hps.encoder_length], name='input')
        self.kp = tf.placeholder(tf.float32, name='keep_prob')
        self.y = tf.placeholder(tf.int32, [None, hps.decoder_length], name='target')
        self.batch_size_tensor = tf.shape(self.x)[0]

    def _add_embedding(self):
        vocab = self._vocab
        embedding_dim = self._hps.embedding_dim
        with tf.variable_scope('embedding') as scope:
            word_matrix = tf.get_variable('embedding', [vocab.size() - (vocab.num_symbols + vocab.num_unks), embedding_dim], dtype=tf.float32, trainable=False)
            symbol_matrix = tf.get_variable('symbols', [vocab.num_symbols + vocab.num_unks, embedding_dim], dtype=tf.float32)
            self.embedding_matrix = tf.concat([symbol_matrix, word_matrix], axis=0)

    def _add_encoder(self, encoder_inputs):
        with tf.variable_scope('encoder'):
            fw_cell = tf.contrib.rnn.LSTMCell(self._hps.hidden_dim)
            fw_cell = tf.contrib.rnn.DropoutWrapper(
                fw_cell,
                #input_keep_prob=self.kp,
                #output_keep_prob=self.kp,
                state_keep_prob=self.kp
            )
            bw_cell = tf.contrib.rnn.LSTMCell(self._hps.hidden_dim)
            bw_cell = tf.contrib.rnn.DropoutWrapper(
                bw_cell, 
                #input_keep_prob=self.kp,
                #output_keep_prob=self.kp,
                state_keep_prob=self.kp
            )
            (encoder_outputs, (fw_state, bw_state)) = tf.nn.bidirectional_dynamic_rnn(
                fw_cell,
                bw_cell, 
                encoder_inputs,
                dtype=tf.float32,
            )
            encoder_outputs = tf.concat(values=encoder_outputs, axis=2)
            return encoder_outputs, fw_state, bw_state

    def _reduce_state(self, fw_state, bw_state):
        with tf.variable_scope('reduce') as scope:
            old_c = tf.concat(values=[fw_state.c, bw_state.c], axis=1)
            old_h = tf.concat(values=[fw_state.h, bw_state.h], axis=1)
            new_c = tf.layers.dense(
                old_c,
                units=self._hps.hidden_dim,
                activation=tf.nn.relu,
                name='reduce_c'
            )
            new_h = tf.layers.dense(
                old_h,
                units=self._hps.hidden_dim,
                activation=tf.nn.relu,
                name='reduce_h'
            )
            return tf.contrib.rnn.LSTMStateTuple(new_c, new_h)

    def _add_decoder(self, inputs, initial_state, encoder_states, feed_previous=True):
        with tf.variable_scope('decoder') as scope:
            cell = tf.contrib.rnn.LSTMCell(self._hps.hidden_dim)
            cell = tf.contrib.rnn.DropoutWrapper(
                cell,
                #input_keep_prob=self.kp,
                #output_keep_prob=self.kp,
                state_keep_prob=self.kp
            )
            return self.PointerDecoder(inputs, initial_state, encoder_states, cell, feed_previous)

    def _add_train_op(self):
        hps = self._hps
        global_step = tf.Variable(0, trainable=False, name='global_step')
        #self._lr = tf.train.exponential_decay(
        #    learning_rate=hps.lr, 
        #    global_step=global_step, 
        #    decay_steps=hps.decay_steps,
        #    decay_rate=hps.decay_rate,
        #    name='learing_rate'
        #)
        #self._nll_opt = tf.train.GradientDescentOptimizer(learning_rate=hps.lr).minimize(self._log_loss, global_step=global_step)
        self._nll_opt = tf.train.RMSPropOptimizer(learning_rate=hps.lr).minimize(self._log_loss, global_step=global_step)
        #self._nll_opt = tf.train.AdagradOptimizer(learning_rate=hps.lr).minimize(self._log_loss, global_step=global_step)
        self._coverage_opt = tf.train.AdagradOptimizer(learning_rate=hps.lr).minimize(self._coverage_loss, global_step=global_step)

    def build_graph(self):
        hps = self._hps
        vocab = self._vocab
        with tf.variable_scope('seq2seq') as scope:
            self._add_placeholder()
            self._add_embedding()
            encoder_inputs = tf.nn.embedding_lookup(self.embedding_matrix, self.x)
            encoder_outputs, fw_state, bw_state = self._add_encoder(encoder_inputs)
            new_state = self._reduce_state(fw_state, bw_state)
            decoder_index_inputs = [tf.ones([self.batch_size_tensor], dtype=tf.int32)] * vocab.word2idx['<BOS>'] + tf.unstack(self.y, axis=1)[:-1]
            train_outputs_list, coverage_loss = self._add_decoder(decoder_index_inputs, new_state, encoder_outputs, False)
            self.train_logits = tf.stack(train_outputs_list, axis=1)
            self.train_predicts = tf.argmax(self.train_logits, axis=-1)
            scope.reuse_variables()
            infer_outputs_list, _ = self._add_decoder(decoder_index_inputs, new_state, encoder_outputs, True)
            self.infer_logits = tf.stack(infer_outputs_list, axis=1)
            self.infer_predicts = tf.argmax(self.infer_logits, axis=-1)

        with tf.variable_scope('loss') as scope:
            mask = tf.cast(tf.sign(self.y), dtype=tf.float32)
            self._log_loss = tf.contrib.seq2seq.sequence_loss(
                logits=self.train_logits,
                targets=self.y,
                weights=mask,
            )
            self._valid_log_loss = tf.contrib.seq2seq.sequence_loss(
                logits=self.infer_logits,
                targets=self.y,
                weights=mask,
            )
            coverage_loss = coverage_loss * tf.expand_dims(mask, axis=2)
            self._coverage_loss = tf.reduce_mean(coverage_loss)

        with tf.variable_scope('training_opt') as scope:
            self._add_train_op()

    def load_embedding(self, npy_path='/home/jjery2243542/datasets/summary/structured/15673_100_20/glove.npy'):
        glove_vectors = np.loadtxt(npy_path)
        self.sess.run(self.embedding_matrix, feed_dict={self.embedding_matrix:glove_vectors})
        print('pretrain vectors loaded')

    def save_model(self, path, epoch):
        self.saver.save(self.sess, path, global_step=epoch)
        
    def train(self, data_generator, log_file_path, model_path, valid_partial=False):
        # init iterator
        print('start training...')
        # calculate time
        start_time = time.time()
        # create log df
        with open(log_file_path, 'w') as f_log:
            f_log.write('epoch,coverage,train_loss,val_loss\n')
            for epoch in range(self._hps.nll_epochs + self._hps.coverage_epochs):
                coverage = True if epoch > self._hps.nll_epochs else False
                total_loss = 0.
                train_iter = data_generator.make_batch(batch_size=self._hps.batch_size, dataset_type='train')
                for i, (batch_x, batch_y) in enumerate(train_iter):
                    loss = self.train_step(batch_x, batch_y, coverage=coverage)
                    total_loss += loss
                    print('epoch [%02d/%02d], step [%06d/%06d], coverage=%r, loss: %.4f, time: %05d\r' % (epoch+1, self._hps.nll_epochs + self._hps.coverage_epochs, i+1, data_generator.size('train')/self._hps.batch_size, coverage, total_loss / (i + 1), time.time() - start_time), end='')
                if valid_partial:
                    valid_iter = data_generator.make_batch(num_datapoints=10000, batch_size=64, dataset_type='valid')
                else:
                    valid_iter = data_generator.make_batch(batch_size=64, dataset_type='valid')
                val_loss = self.valid(valid_iter)
                print('\nepoch [%02d/%02d], train_loss: %.4f, val_loss: %.4f, time: %.02f' % (epoch + 1, self._hps.nll_epochs + self._hps.coverage_epochs, total_loss / (i + 1), val_loss, time.time() - start_time))
                # write to log file
                f_log.write('%02d,%r,%.4f,%.4f\n' % (epoch, False, total_loss / (i + 1), val_loss))
                # save to model
                self.save_model(model_path, epoch)

    def valid(self, iterator, dataset_type='valid'):
        vocab = self._vocab
        total_loss = 0. 
        i = 0
        for batch_x, batch_y in iterator:
            loss = self.valid_step(batch_x, batch_y)
            total_loss += loss
            i += 1
        avg_loss = total_loss / (i + 1)
        return avg_loss 

    def init(self, npy_path=None, pretrain=True):
        self.sess.run(tf.global_variables_initializer())
        if pretrain:
            # load pretrain glove vector
            self.load_embedding(npy_path)

    def valid_step(self, batch_x, batch_y):
        loss = self.sess.run(
            self._valid_log_loss,
            feed_dict={self.x:batch_x, self.y:batch_y, self.kp:1.0}
        )
        return loss

    def train_step(self, batch_x, batch_y, coverage=False):
        if not coverage:
            _, loss = self.sess.run(
                [self._nll_opt, self._log_loss], 
                feed_dict={self.x:batch_x, self.y:batch_y, self.kp:self._hps.keep_prob}
            )
        else:
            _, loss = self.sess.run(
                [self._coverage_opt, self._coverage_loss], 
                feed_dict={self.x:batch_x, self.y:batch_y, self.kp:self._hps.keep_prob}
            )
        return loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-hps_path')
    parser.add_argument('-dataset_path', default='/home/jjery2243542/datasets/summary/structured/15673_100_20/giga_80_15.hdf5')
    parser.add_argument('--pretrain', action='store_false')
    parser.add_argument('-npy_path', default='/home/jjery2243542/datasets/summary/structured/15673_100_20/glove.npy')
    parser.add_argument('-log_file_path', default='./log.txt')
    parser.add_argument('-model_path', default='./model/model.ckpt')
    parser.add_argument('--valid_partial', action='store_false')
    args = parser.parse_args()
    if args.hps_path:
        hps = Hps()
        hps.load(args.args.hps_path)
        hps_tuple = hps.get_tuple()
    else:
        hps = Hps()
        hps_tuple = hps.get_tuple()
    print(hps_tuple)
    vocab = Vocab()
    data_generator = DataGenerator(args.dataset_path)
    model = PointerModel(hps_tuple, vocab)
    if args.pretrain:
        model.init(npy_path=args.npy_path, pretrain=True)
    model.train(
        data_generator=data_generator, 
        log_file_path=args.log_file_path, 
        model_path=args.model_path,
        valid_partial=args.valid_partial
    )
