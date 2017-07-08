import tensorflow as tf

class PointerModel(object):
    def __init__(self, hps, vocab):
        self._hps = hps
        self._vocab = vocab

    def _add_placeholder(self):
        hps = self._hps
        self.x = tf.placeholder(tf.int32, [hps.batch_size, hps.encoder_length], name='input')
        self.kp = tf.placeholder(tf.float32, name='keep_prob')
        self.y = tf.placeholder(tf.int32, [hps.batch_size, hps.encoder_length], name='target')
    
    def _add_encoder(self, encoder_inputs):
        with tf.variable_scope('encoder'):
            fw_cell = tf.contrib.rnn.LSTMCell(self._hps.hidden_dim)
            fw_cell = tf.contrib.DropoutWrapper(
                fw_cell, 
                input_keep_prob=self.kp,
                output_keep_prob=self.kp,
                state_keep_prob=self.kp,
            )
            bw_cell = tf.contrib.rnn.LSTMCell(self._hps.hidden_dim)
            bw_cell = tf.contrib.DropoutWrapper(
                bw_cell, 
                input_keep_prob=self.kp,
                output_keep_prob=self.kp,
                state_keep_prob=self.kp,
            )
            (encoder_outputs, (fw_state, bw_state)) = tf.nn.bidirectional_dynamic_rnn(
                cell_fw,
                cell_bw, 
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

