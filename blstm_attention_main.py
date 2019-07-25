dim_input = 44
sequence_len = 128
learning_rate = 5e-5
min_learning_rate = 1e-5
batch_size = 50
step = 1
training_iters = 100000
batch_size = 128
display_step = 10
n_hidden = 350
n_steps=128
file_path = './spoken_numbers_pcm/'

import warnings
import numpy as np
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from sklearn.utils import compute_class_weight
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Bidirectional, LeakyReLU
from keras.optimizers import Adamax
from keras.layers import Layer
from keras import backend as K
from keras import initializers, regularizers, constraints
from sklearn.model_selection import train_test_split



"""
Bidirectional LSTM neural network with attention
"""


def dot_product(x, kernel):
    """
    Wrapper for dot product operation, in order to be compatible with both Theano and Tensorflow
    Args:
        x (): input
        kernel (): weights
    Returns:
    """
    if K.backend() == 'tensorflow':
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)


class AttentionWithContext(Layer):
    """
    Attention operation, with a context/query vector, for temporal data.
    Supports Masking.
    follows these equations:
    (1) u_t = tanh(W h_t + b)
    (2) \alpha_t = \frac{exp(u^T u)}{\sum_t(exp(u_t^T u))}, this is the attention weight
    (3) v_t = \alpha_t * h_t, v in time t
    # Input shape
        3D tensor with shape: `(samples, steps, features)`.
    # Output shape
        3D tensor with shape: `(samples, steps, features)`.
    """

    def __init__(self,
                 W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 bias=True, **kwargs):

        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(AttentionWithContext, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

        self.u = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.u_regularizer,
                                 constraint=self.u_constraint)

        super(AttentionWithContext, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        uit = dot_product(x, self.W)

        if self.bias:
            uit += self.b

        uit = K.tanh(uit)
        ait = dot_product(uit, self.u)

        a = K.exp(ait)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero and this results in NaN's.
        # Should add a small epsilon as the workaround
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a

        return weighted_input

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1], input_shape[2]


class Addition(Layer):
    """
    This layer is supposed to add of all activation weight.
    We split this from AttentionWithContext to help us getting the activation weights
    follows this equation:
    (1) v = \sum_t(\alpha_t * h_t)
    # Input shape
        3D tensor with shape: `(samples, steps, features)`.
    # Output shape
        2D tensor with shape: `(samples, features)`.
    """

    def __init__(self, **kwargs):
        super(Addition, self).__init__(**kwargs)

    def build(self, input_shape):
        self.output_dim = input_shape[-1]
        super(Addition, self).build(input_shape)

    def call(self, x):
        return K.sum(x, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)
class BLSTM_Attention:
    
    def __init__(self, x_train, y_train,x_test,y_test,batch_size,epochs, dropout,lr,name):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.name = name
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.class_weight = compute_class_weight(class_weight='balanced', classes=np.arange(10), y=y_train.argmax(axis=1))
        model = Sequential()
        model.add(Bidirectional(LSTM(300, return_sequences=True), input_shape=(n_steps,dim_input)))
        model.add(AttentionWithContext())
        model.add(Addition())
        model.add(Dense(300))
        model.add(LeakyReLU())
        model.add(Dropout(dropout))
        model.add(Dense(300))
        model.add(LeakyReLU())
        model.add(Dropout(dropout))
        model.add(Dense(10, activation='softmax'))
        # Lower learning rate to prevent divergence
        adamax = Adamax(self.lr)
        model.compile(adamax, 'categorical_crossentropy', metrics=['accuracy'])
        self.model = model

    """
    Trains model
    """

    def train(self):
        self.model.fit(self.x_train, self.y_train, batch_size=self.batch_size, epochs=self.epochs,
                       class_weight=self.class_weight)
        self.model.save_weights(self.name + "_model.pkl")

    """
    Tests accuracy of model
    Loads weights from file if no weights are attached to model object
    """

    def test(self):
        self.model.load_weights(self.name + "_model.pkl")
        values = self.model.evaluate(self.x_test, to_categorical(self.y_test), batch_size=self.batch_size)
        print("Accuracy: ", values[1])
        predictions = (self.model.predict(self.x_test, batch_size=len(self.x_test)))
        
        y_pre = np.argmax(predictions, axis=1)
        y_true = self.y_test
        print(np.sum(np.array(y_pre==y_true).astype('float'))/len(y_pre))
#         target_names = ['0', '1', '2','3','4','5','6','7','8','9']
#         print(classification_report(y_true, y_pred, target_names=target_names))
            
        
        
        
        
#         #predictions = predictions.argmax(axis=1)
#         tn, fp, fn, tp = confusion_matrix(np.argmax(self.y_test, axis=1), np.argmax(predictions, axis=1)).ravel()
#         print('False positive rate(FP): ', fp / (fp + tn))
#         print('False negative rate(FN): ', fn / (fn + tp))
#         recall = tp / (tp + fn)
#         print('Recall: ', recall)
#         precision = tp / (tp + fp)
#         print('Precision: ', precision)
#         print('F1 score: ', (2 * precision * recall) / (precision + recall))

model = BLSTM_Attention(train_data_x,train_data_y,test_data_x,test_data_y,batch_size,1,0.2,learning_rate,'blstm_attention')
model.train()
model.test()
