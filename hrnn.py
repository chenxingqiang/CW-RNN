"""
Hierarchical RNN
"""
exec(open("preprocess.py").read())
exec(open("helper_function.py").read())
from keras.models import Model
from keras.layers import Input, Dense, TimeDistributed, GRU, Bidirectional, Masking
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
ecp = importr('ecp')
pandas2ri.activate()


class HRNN():
   
    def __init__(self, sub_hidden, seq_hidden, max_length_sub, dim):
        self.max_length_sub = max_length_sub
        self.dim = dim
        self.x = Input(shape=(None, max_length_sub, dim))
        self.masked_x = TimeDistributed(Masking(mask_value=0.))(self.x)
        self.sub_seq = TimeDistributed(Bidirectional(GRU(sub_hidden, recurrent_dropout=0.5)))(self.masked_x)
        self.masked_sub_seq = Masking(mask_value=0.)(self.sub_seq)
        self.seq = GRU(seq_hidden, recurrent_dropout=0.5)(self.masked_sub_seq)
        self.prediction = Dense(1, activation='sigmoid')(self.seq)

        self.model = Model(inputs=self.x, outputs=self.prediction)
        self.model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

    
    def fit(self, x_train, y_train, batch_size, epochs):
        self.batch_size = batch_size
        self.epochs = epochs
        self.model.fit(x_train, y_train, epochs=self.epochs, batch_size=self.batch_size)
        

    def predict(self, i_test, min_size, w_size):   
        self.min_size = min_size
        self.max_length_test = np.max(sizes2[i_test])
        self.w_size = w_size
        self.risk = np.zeros((i_test.size, self.max_length_test))
        for i in range(i_test.size):
            index = i_test[i]
            temp = timeSeriesCon[index]
            # print("i={}\n".format(i))
            for j in range(min_size, temp.shape[0]):
                start = max(j-w_size,0)
                temp1 = temp[start:j,:]
                cps = ecp.e_divisive(temp1, min_size=min_size)
                temp1_sub = divideSeq([cps], [temp1])
                temp1_sub_pad = toFixedLength(temp1_sub, self.max_length_sub)
                temp1_sub_pad = np.expand_dims(temp1_sub_pad, axis=0)
                self.risk[i, j+self.max_length_test-sizes2[index]] = self.model.predict(temp1_sub_pad)[0,0]
        return self.risk
    
    
    
    
    
    
    
