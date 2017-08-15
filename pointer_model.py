import tensorflow as tf
from utils import Hps
from utils import Vocab
import time

class PointerModel(object):
    def __init__(self, hps, vocab):
        self._hps = hps
        self._vocab = vocab
        self.build_graph()
        self.sess = tf.Session()
        
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
            attn_coverage = tf.zeros([encoder_states.get_shape()[0].value, encoder_states.get_shape()[1].value], dtype=tf.float32)
            # do attention for the first time
            attn_weights, attn_context = attention(state.h, attn_coverage)
            for i in range(len(decoder_inputs)):
                if i > 0:
                    scope.reuse_variables()
                elif i > 0 and feed_previous:
                    last_output_id = tf.argmax(decoder_outputs[-1], axis=-1)
                    inp = tf.nn.embedding_lookup(self.embedding_matrix, last_output_id, axis=-1)
                else:
                    inp = decoder_inputs[i]
                cell_input = input_projection(inp, attn_context)
                cell_output, state = cell(cell_input, state)
                attn_weights, attn_context = attention(state.h, attn_coverage)
                coverage_loss_list.append(tf.minimum(attn_coverage, attn_weights))
                attn_coverage += attn_weights

                p_gen = gen_layer(inp, state, attn_context)
                output_t = p_gen * get_vocab_distr(cell_output, attn_context) + (1 - p_gen) * get_pointer_distr(attn_weights)
                decoder_outputs.append(output_t)
            coverage_loss = tf.stack(coverage_loss_list, axis=1)
            return decoder_outputs, coverage_loss

    def _add_placeholder(self):
        hps = self._hps
        self.x = tf.placeholder(tf.int32, [hps.batch_size, hps.encoder_length], name='input')
        self.kp = tf.placeholder(tf.float32, name='keep_prob')
        self.y = tf.placeholder(tf.int32, [hps.batch_size, hps.decoder_length], name='target')

    def _add_embedding(self):
        with tf.variable_scope('embedding') as scope:
            self.embedding_matrix = tf.get_variable('embedding', [self._vocab.size(), self._hps.embedding_dim], dtype=tf.float32, trainable=False)

    def _add_encoder(self, encoder_inputs):
        with tf.variable_scope('encoder'):
            fw_cell = tf.contrib.rnn.LSTMCell(self._hps.hidden_dim)
            fw_cell = tf.contrib.rnn.DropoutWrapper(
                fw_cell,
                input_keep_prob=self.kp,
                output_keep_prob=self.kp,
                state_keep_prob=self.kp
            )
            bw_cell = tf.contrib.rnn.LSTMCell(self._hps.hidden_dim)
            bw_cell = tf.contrib.rnn.DropoutWrapper(
                bw_cell, 
                input_keep_prob=self.kp,
                output_keep_prob=self.kp,
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
                #activation=tf.nn.relu,
                name='reduce_c'
            )
            new_h = tf.layers.dense(
                old_h,
                units=self._hps.hidden_dim,
                #activation=tf.nn.relu,
                name='reduce_h'
            )
            return tf.contrib.rnn.LSTMStateTuple(new_c, new_h)

    def _add_decoder(self, inputs, initial_state, encoder_states, feed_previous=True):
        with tf.variable_scope('decoder') as scope:
            cell = tf.contrib.rnn.LSTMCell(self._hps.hidden_dim)
            cell = tf.contrib.rnn.DropoutWrapper(
                cell,
                input_keep_prob=self.kp,
                output_keep_prob=self.kp,
                state_keep_prob=self.kp
            )
            return self.PointerDecoder(inputs, initial_state, encoder_states, cell, feed_previous)

    def _add_train_op(self):
        hps = self._hps
        global_step = tf.Variable(0, trainable=False, name='global_step')
        self._lr = tf.train.inverse_time_decay(
            learning_rate=hps.lr, 
            global_step=global_step, 
            decay_steps=hps.decay_steps,
            decay_rate=hps.decay_rate,
            name='learing_rate'
        )
        self._nll_opt = tf.train.GradientDescentOptimizer(learning_rate=self._lr).minimize(self._log_loss, global_step=global_step)
        self._coverage_opt = tf.train.GradientDescentOptimizer(learning_rate=self._lr).minimize(self._coverage_loss, global_step=global_step)

    def build_graph(self):
        hps = self._hps
        vocab = self._vocab
        with tf.variable_scope('seq2seq') as scope:
            self._add_placeholder()
            self._add_embedding()
            encoder_inputs = tf.nn.embedding_lookup(self.embedding_matrix, self.x)
            encoder_outputs, fw_state, bw_state = self._add_encoder(encoder_inputs)
            new_state = self._reduce_state(fw_state, bw_state)
            decoder_index_inputs = [tf.ones([hps.batch_size], dtype=tf.int32)] * vocab.word2idx['<BOS>'] + tf.unstack(self.y, axis=1)
            decoder_embedding_inputs = [tf.nn.embedding_lookup(self.embedding_matrix, x) for x in decoder_index_inputs[:-1]]
            train_outputs_list, coverage_loss = self._add_decoder(decoder_embedding_inputs, new_state, encoder_outputs, False)
            self.train_logits = tf.stack(train_outputs_list, axis=1)
            scope.reuse_variables()
            infer_outputs_list, _ = self._add_decoder(decoder_embedding_inputs, new_state, encoder_outputs, True)
            self.infer_logits = tf.stack(infer_outputs_list, axis=1)
            self.infer_predicts = tf.argmax(self.infer_logits, axis=-1)

        with tf.variable_scope('loss') as scope:
            mask = tf.cast(tf.sign(self.y), dtype=tf.float32)
            self._log_loss = tf.contrib.seq2seq.sequence_loss(
                logits=self.train_logits,
                targets=self.y,
                weights=mask,
            )
            coverage_loss = coverage_loss * tf.expand_dims(mask, axis=2)
            self._coverage_loss = tf.reduce_mean(coverage_loss)

        with tf.variable_scope('training_opt') as scope:
            self._add_train_op()

    def load_embedding(self, sess):
        sess.run()

    def train(self, sess, data_generator, log_file_path=None):
        start_time = time.time()
        print('NLL section')
        for epoch in range(self._hps.nll_epochs):
            total_loss = 0.
            for i, (batch_x, batch_y) in enumerate(data_generator):
                loss, lr = self.train_step(sess, batch_x, batch_y, coverage=False)
                total_loss += loss
                if (i + 1) % 1000 == 0:
                    print('\nepoch [%02d/%02d], step [%06d/%06d], lr=%.4f, loss: %.4f, time: %05d\r' % (epoch + 1, self._hps.nll_epochs, lr, total_loss / (i + 1), time.time() - start_time), end='')
            if log_file_path:
                with open(log_file_path, 'a') a f_log:
                    f_log.write('epoch: %02d, avg_train_loss: %.4f' % (epoch + 1, total_loss / (i + 1)))

    def valid(self, sess, data_generator):
        for batch_x, batch_y in data_generator:
            loss = self.

    def init(self, sess, pretrain=True):
        sess.run(tf.global_variables_initializer())
        if pretrain:
            # load pretrain glove vector
    def valid_step()
    def train_step(self, batch_x, batch_y, coverage=False):
        if not coverage:
            _, loss, lr = self.sess.run(
                [self._nll_opt, self._log_loss, self._lr], 
                feed_dict={self.x:batch_x, self.y:batch_y, self.kp:self._hps.keep_prob}
            )
        else:
            _, loss, lr = self.sess.run(
                [self._coverage_opt, self._coverage_loss, self._lr], 
                feed_dict={self.x:batch_x, self.y:batch_y, self.kp:self._hps.keep_prob}
            )
        return loss, lr

if __name__ == '__main__':
    hps = Hps()
    vocab = Vocab()
