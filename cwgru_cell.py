'''
This RNN cell is a combination of Clockwork RNN and Gated Recurrent Unit (GRU), we name it "CWGRU"
'''
import tensorflow as tf
from tensorflow.python.ops import rnn_cell_impl
import numpy as np

class ClockworkGRUCell(rnn_cell_impl.RNNCell):
    
    
    def __init__(self, periods, group_size, variant=0, activation=tf.tanh, reuse=None, kernel_initializer=None,
               bias_initializer=None):
        self._kernel_initializer = kernel_initializer
        self._bias_initializer = bias_initializer
        self._num_units = len(periods) * group_size
        super(ClockworkGRUCell, self).__init__(self._num_units)
        self._activation = activation
        self._periods = periods
        self._group_size = group_size
        self.clockwork_mask = tf.constant(self._make_block_ltriangular(
                np.ones([self._num_units, self._num_units]), 
                self._group_size, variant), dtype=tf.float32, name="mask")
        
    
    def call(self, inputs, state):
        self._timestep = tf.cast(state[1], tf.int32)
        state = state[0]
        # Update cell state
        new_state = self._update_state(inputs, state)
        new_timestep = tf.cast(tf.add(self._timestep, 1), tf.float32)
        new_timestep.set_shape([inputs.get_shape()[0], 1])
        return new_state, (new_state, new_timestep)
    
    
    @property
    def state_size(self):
        return self._num_units, 1
    
    @property
    def output_size(self):
        return self._num_units
    
    def _update_state(self, inputs, state):
        # Weight and bias initializers
        initializer_kernel = tf.contrib.keras.initializers.glorot_normal()
        initializer_rec_kernel = tf.contrib.keras.initializers.Orthogonal()
        initializer_bias    = tf.constant_initializer(0.0)
        
        active_index = self._compute_active_index()
        
        input_size = inputs.get_shape()[1]

        with tf.variable_scope("forget_gate"):
            hidden_WH_u = tf.get_variable("WH", shape=[self._num_units, self._num_units], initializer=initializer_rec_kernel)
            hidden_WH_u = tf.multiply(hidden_WH_u, self.clockwork_mask)
            hidden_WI_u = tf.get_variable("WI", shape=[input_size, self._num_units], initializer=initializer_kernel)
            hidden_W_u = tf.concat([hidden_WI_u, hidden_WH_u], axis=0)
            hidden_b_u = tf.get_variable("b", shape=[self._num_units], initializer=initializer_bias)
        
        with tf.variable_scope("input_gate"):
            hidden_WH_r = tf.get_variable("WH", shape=[self._num_units, self._num_units], initializer=initializer_rec_kernel)
            hidden_WH_r = tf.multiply(hidden_WH_r, self.clockwork_mask)
            hidden_WI_r = tf.get_variable("WI", shape=[input_size, self._num_units], initializer=initializer_kernel)
            hidden_W_r = tf.concat([hidden_WI_r, hidden_WH_r], axis=0)
            hidden_b_r = tf.get_variable("b", shape=[self._num_units], initializer=initializer_bias)
        
        with tf.variable_scope("candidate"):
            hidden_WH_c = tf.get_variable("WH", shape=[self._num_units, self._num_units], initializer=initializer_rec_kernel)
            hidden_WH_c = tf.multiply(hidden_WH_c, self.clockwork_mask)
            hidden_WI_c = tf.get_variable("WI", shape=[input_size, self._num_units], initializer=initializer_kernel)
            hidden_b_c = tf.get_variable("b", shape=[self._num_units], initializer=initializer_bias)
        
        with tf.variable_scope("clockwork_gru_cell"):
            input_ = tf.concat([inputs, state], axis=1)
            W_forget = tf.matmul(input_,  tf.slice(hidden_W_u, [0, 0], [-1, active_index]))
            W_forget = tf.nn.sigmoid(tf.nn.bias_add(W_forget, tf.slice(hidden_b_u, [0], [active_index]), name="W_forget"))
            
            W_input = tf.matmul(input_, tf.slice(hidden_W_r, [0, 0], [-1, active_index]))
            W_input = tf.nn.sigmoid(tf.nn.bias_add(W_input, tf.slice(hidden_b_r, [0], [active_index]), name="W_input"))
            
            state_sliced = tf.slice(state, [0, 0], [-1, active_index])
            state_c = W_input * state_sliced
            W_candidate = tf.matmul(inputs, tf.slice(hidden_WI_c, [0, 0], [-1, active_index])) + tf.matmul(state_c, tf.slice(hidden_WH_c, [0, 0], [active_index, active_index]))
            W_candidate = self._activation(tf.nn.bias_add(W_candidate, tf.slice(hidden_b_c, [0], [active_index]), name="W_candidate"))
            
            y_update = tf.add(W_forget*state_sliced, (1-W_forget)*W_candidate, name="state_update")
            
            # Copy the updates to the cell state
            new_state = tf.concat(
                axis=1, values=[y_update, tf.slice(state, [0, active_index], [-1,-1])])
        
            new_state.set_shape([inputs.get_shape()[0], self._num_units])
            
        return new_state
            

    def _compute_active_index(self):
        group_index = 1
        for i in range(len(self._periods)):
            # Check if (t MOD T_i == 0)
            group_index = tf.where(tf.reduce_all(tf.equal(tf.mod(self._timestep, self._periods[i]),0)), i+1, group_index)
        return tf.multiply(self._group_size, group_index, name="activation_index")


    @staticmethod
    def _make_block_ltriangular(m, group_size, variant=0):
        assert m.shape[0] == m.shape[1]
        assert m.shape[0] % group_size == 0
        if(variant==0): # original one in paper
            for i in range(m.shape[0]//group_size-1):
                m[i*group_size:(i+1)*group_size, (i+1)*group_size:] = 0
        elif(variant==1): # connect units of adjacent module
            for i in range(m.shape[0]//group_size-1):
                m[i*group_size:(i+1)*group_size, (i+1)*group_size:] = 0
            for i in range(m.shape[0]//group_size-2):
                m[(i+2)*group_size:,i*group_size:(i+1)*group_size] = 0
        elif(variant==2):
            pass
        return m
