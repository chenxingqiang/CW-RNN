"""
This is a Clockwork RNN that can adaptively allocate model resources to different resolutions based on the input. Each hidden unit is "softly" assigned to each group with a leaky parameter. In order to be adaptive, it uses a two layer feed-forward network to determine the extent to which each neuron belongs to a group at each timestep. 
Additional argument:
    length_limit: the maximum length of the input timeseries
"""

import tensorflow as tf
import numpy as np


class AR_CWRNNCell(tf.contrib.rnn.BasicRNNCell):
    
    def __init__(self, periods, num_units, length_limit, activation=tf.tanh, reuse=None):
        self._num_units = num_units
        super(AR_CWRNNCell, self).__init__(self._num_units)
        self._activation = activation
        self._nb_group = len(periods)
        self._periods = periods
        self.keys = tf.constant(self._generate(length_limit), dtype=tf.int32)
        
    
    def call(self, inputs, state):
        self._timestep = tf.cast(state[1], tf.int32)
        state = state[0]
        # Update cell state
        new_state = self._update_state(inputs, state)
        new_timestep = tf.cast(tf.add(self._timestep, 1), tf.float32)
        new_timestep.set_shape([inputs.get_shape()[0], 1])
        return (new_state, self.new_output), (new_state, new_timestep)
    
    @property
    def state_size(self):
        return self._num_units, 1
    
    @property
    def output_size(self):
        return self._num_units, self._nb_group

    def _update_state(self, inputs, state):
        self.batch_size = tf.shape(inputs)[0]
        self.dim = inputs.get_shape().as_list()[1]

        # Weight and bias initializers
        self.initializer_kernel = tf.contrib.keras.initializers.glorot_normal()
        self.initializer_rec_kernel = tf.contrib.keras.initializers.Orthogonal()
        self.initializer_bias    = tf.constant_initializer(0.0)
        
        activation_weights = self._compute_active_index(inputs, state)
        
        with tf.variable_scope("input"):
            input_W = tf.get_variable("W", shape=[inputs.get_shape()[1], self._num_units], initializer=self.initializer_kernel)

        with tf.variable_scope("hidden"):
            hidden_W = tf.get_variable("W", shape=[self._num_units, self._num_units], initializer=self.initializer_rec_kernel)
            hidden_b = tf.get_variable("b", shape=[self._num_units], initializer=self.initializer_bias)
        with tf.variable_scope("clockwork_cell"):
            WI_x = tf.matmul(inputs, input_W, name="WI_x")
            WH_y = tf.matmul(state, hidden_W)
            WH_y = tf.nn.bias_add(WH_y, hidden_b, name="WH_y")
            # Compute y_t = (...) and update the cell state
            y_update = tf.add(WH_y, WI_x, name="state_update")
            y_update = self._activation(y_update)
            
            new_state = tf.multiply(activation_weights, y_update) + tf.multiply(1-activation_weights, state)
            
            # Copy the updates to the cell state
            new_state.set_shape([inputs.get_shape()[0], self._num_units])
        
        return new_state
    
        
    
    def _compute_active_index(self, inputs, state):
        
#        # decide activation group using pre-fixed periods
#        predict_group = 1
#        for i in range(len(self._periods)):
#            # Check if (t MOD T_i == 0)
#            predict_group = tf.where(tf.reduce_all(tf.equal(tf.mod(self._timestep, self._periods[i]),0)), i+1, predict_group)
        predict_group = tf.squeeze(tf.gather(self.keys, tf.gather(self._timestep,0)))
        # decide allocation of resources
        predict_groups = tf.tile(tf.reshape(predict_group,[1]), [self.batch_size]) #shape: batch_size,
        embedding_size = 10
        predict_groups = tf.contrib.keras.layers.Embedding(self._nb_group, embedding_size)(predict_groups-1)
        l1_size = 4
        with tf.variable_scope("compute_active_weight_l1"):
            active_W1 = tf.get_variable("W", shape=[self._num_units+self.dim+embedding_size, l1_size], initializer=self.initializer_kernel)
            active_b1 = tf.get_variable("b", shape=[l1_size], initializer=self.initializer_bias)
        with tf.variable_scope("compute_active_weight_l2"):
            active_W2 = tf.get_variable("W", shape=[l1_size, self._nb_group*self._num_units], initializer=self.initializer_kernel)
            active_b2 = tf.get_variable("b", shape=[self._nb_group*self._num_units], initializer=self.initializer_bias)
        self.active_W1 = active_W1
        self.active_W2 = active_W2
        active_logits1 = tf.nn.bias_add(tf.matmul(tf.concat([inputs,state,predict_groups],axis=1),active_W1), active_b1)
        active_logits = tf.nn.bias_add(tf.matmul(active_logits1, active_W2), active_b2)
        active_logits = tf.reshape(active_logits, [-1, self._num_units, self._nb_group])
        # dropout
        #active_logits = tf.nn.drop_out(active_logits, 0.3)
        # sharpen the distribution
        alloc_resources = tf.nn.softmax(active_logits) #shape: batch_size * num_units * nb_group
        self.new_output = tf.reduce_sum(alloc_resources, axis=1)/self._num_units
        alloc_resources_tr = tf.transpose(alloc_resources, [2,0,1])
        # decide activation weights
        activation_weights = tf.reduce_sum(tf.gather(alloc_resources_tr, tf.range(predict_group)), axis=0) # shape: batch_size * num_units 
        #self.new_ar = activation_weights
        return activation_weights
    
    def _generate(self, n):
        output = []
        for i in range(n):
            if(np.mod(i,32)==0):
                temp=6
            elif(np.mod(i,16)==0):
                temp=5
            elif(np.mod(i,8)==0):
                temp=4
            elif(np.mod(i,4)==0):
                temp=3
            elif(np.mod(i,2)==0):
                temp=2
            elif(np.mod(i,1)==0):
                temp=1
            output.append(temp)
        return output
        
