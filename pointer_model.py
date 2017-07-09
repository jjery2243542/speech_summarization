import tensorflow as tf
from utils import Hps
from utils import Vocab

class PointerModel(object):
    def __init__(self, hps, vocab):
        self._hps = hps
        self._vocab = vocab
        self.build_graph()
    
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
                output_layer_hidden = tf.layers.dense(output_layer_input, hidden_dim, name='output_projection')
                vocab_distr = tf.layers.dense(output_layer_hidden, vocab_size, activation=tf.nn.softmax, name='output_layer')
                return vocab_distr
                
            decoder_outputs = []
            coverage_loss = []
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
                coverage_loss.append(tf.minimum(attn_coverage, attn_weights))
                attn_coverage += attn_weights

                p_gen = gen_layer(inp, state, attn_context)
                output_t = p_gen * get_vocab_distr(cell_output, attn_context) + (1 - p_gen) * get_pointer_distr(attn_weights)
                decoder_outputs.append(output_t)
            return decoder_outputs, coverage_loss

    def _add_placeholder(self):
        hps = self._hps
        self.x = tf.placeholder(tf.int32, [hps.batch_size, hps.encoder_length], name='input')
        self.kp = tf.placeholder(tf.float32, name='keep_prob')
        self.y = tf.placeholder(tf.int32, [hps.batch_size, hps.encoder_length], name='target')

    def _add_embedding(self):
        with tf.variable_scope('embedding') as scope:
            self.embedding_matrix = tf.get_variable('embedding', [self._vocab.size(), self._hps.embedding_dim], dtype=tf.float32)

    def _add_encoder(self, encoder_inputs):
        with tf.variable_scope('encoder'):
            fw_cell = tf.contrib.rnn.LSTMCell(self._hps.hidden_dim)
            fw_cell = tf.contrib.rnn.DropoutWrapper(
                fw_cell, 
                output_keep_prob=self.kp,
            )
            bw_cell = tf.contrib.rnn.LSTMCell(self._hps.hidden_dim)
            bw_cell = tf.contrib.rnn.DropoutWrapper(
                bw_cell, 
                output_keep_prob=self.kp,
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
            return self.PointerDecoder(inputs, initial_state, encoder_states, cell, feed_previous)

    def build_graph(self):
        hps = self._hps
        vocab = self._vocab
        with tf.variable_scope('seq2seq') as scope:
            self._add_placeholder()
            self._add_embedding()
            encoder_inputs = tf.nn.embedding_lookup(self.embedding_matrix, self.x)
            encoder_inputs = tf.nn.dropout(encoder_inputs, self.kp)
            encoder_outputs, fw_state, bw_state = self._add_encoder(encoder_inputs)
            new_state = self._reduce_state(fw_state, bw_state)
            decoder_index_inputs = [tf.ones([hps.batch_size], dtype=tf.int32)] * vocab.word2idx['<BOS>'] + tf.unstack(self.y, axis=1)
            decoder_embedding_inputs = [tf.nn.embedding_lookup(self.embedding_matrix, x) for x in decoder_index_inputs[:-1]]
            decoder_embedding_inputs = [tf.nn.dropout(x, self.kp) for x in decoder_embedding_inputs]
            train_output_list, coverage_loss_list = self._add_decoder(decoder_embedding_inputs, new_state, encoder_outputs, False)
            scope.reuse_variables()
            infer_output_list, _ = self._add_decoder(decoder_embedding_inputs, new_state, encoder_outputs, True)

if __name__ == '__main__':
    hps = Hps()
    vocab = Vocab()
    Model = PointerModel(hps, vocab)