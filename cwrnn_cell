
import tensorflow as tf
import numpy as np

class ClockworkRNNCell(tf.contrib.rnn.BasicRNNCell):
    '''
    A Clockwork RNN - Koutnik et al. 2014 [arXiv, https://arxiv.org/abs/1402.3511]

    The Clockwork RNN (CW-RNN), in which the hidden layer is partitioned into separate modules,
    each processing inputs at its own temporal granularity, making computations only at its prescribed clock rate.
    Rather than making the standard RNN models more complex, CW-RNN reduces the number of RNN parameters,
    improves the performance significantly in the tasks tested, and speeds up the network evaluation

    '''
    '''
    Arguments for constructing a ClockworkRNNCell:
        periods: the sorted list of periods of different modules, ex: [1,2,4,8,16]
        group_size: the number of units in each module, ex:10, the number of 
        hidden units is therefore len(periods) * group_size
    '''
    
    def __init__(self, periods, group_size, variant=0, activation=tf.tanh, reuse=None):
        self._num_units = len(periods) * group_size
        super(ClockworkRNNCell, self).__init__(self._num_units)
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
        
        print(inputs)
        with tf.variable_scope("input"):
            input_W = tf.get_variable("W", shape=[inputs.get_shape()[1], self._num_units], initializer=initializer_kernel)

        with tf.variable_scope("hidden"):
            hidden_W = tf.get_variable("W", shape=[self._num_units, self._num_units], initializer=initializer_rec_kernel)
            hidden_W = tf.multiply(hidden_W, self.clockwork_mask)
            hidden_b = tf.get_variable("b", shape=[self._num_units], initializer=initializer_bias)
        
        with tf.variable_scope("clockwork_cell"):
            WI_x = tf.matmul(inputs, tf.slice(input_W, [0, 0], [-1, active_index]), name="WI_x")
            WH_y = tf.matmul(state, tf.slice(hidden_W, [0, 0], [-1, active_index]))
            WH_y = tf.nn.bias_add(WH_y, tf.slice(hidden_b, [0], [active_index]), name="WH_y")
            # Compute y_t = (...) and update the cell state
            y_update = tf.add(WH_y, WI_x, name="state_update")
            y_update = self._activation(y_update)
            
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
        elif(variant==2): # use all connections
            pass
        return m
