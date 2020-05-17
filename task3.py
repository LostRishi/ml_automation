#!/usr/bin/env python
# coding: utf-8

# ### Import required Packages

# In[1]:


import tensorflow as tf
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten,MaxPooling2D


# ### Data Preprocessing 

# In[2]:



#download mnist data and split into train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(60000,28,28,1)
X_test = X_test.reshape(10000,28,28,1)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


# ### Helper Functions

# In[3]:


def conv_block(model, n_filters, activation='relu'):
    model.add(Conv2D(n_filters, kernel_size=3, activation=activation))
    model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))
    return model


# In[4]:


def conv_compile(model, optimizer):
    opt = optimizer
    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=['accuracy'])
    return model


# ### Main Function

# In[5]:


def build_model(n_blocks,input_shape, optimizer, activation='relu', epochs = 5):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    while (n_blocks != 0):
        model = conv_block(model,64)
        n_blocks -= 1
    #model = conv_block(model, 64)
    #model = conv_block(model, 64)
    model.add(Flatten())
    model.add(Dense(10, activation="softmax"))
    model.summary()
    model = conv_compile(model, optimizer = optimizer)
    model.fit(X_train, y_train, validation_data = (X_test, y_test), epochs = epochs)
    return model


# In[6]:


x = 0;
custom_model = build_model(0,input_shape = (28,28,1), optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001), epochs = 3)
x+=1;


# ### Accuracy Score

# In[7]:


score = custom_model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[ ]:




