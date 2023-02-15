#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns


# In[8]:


df = pd.read_csv('./breast-cancer (1).csv')
df.head(10)


# In[9]:


df.columns


# In[10]:


df = df.drop(['id' , 'Unnamed: 32'],axis = 1)


# In[11]:


df['diagnosis'] = df['diagnosis'].map({'M':0 , 'B':1}).astype(int)


# In[12]:


df.head()


# In[13]:


from sklearn.model_selection import train_test_split


# In[14]:


X = df.iloc[: , 1:]
y = df.iloc[: , 0]


# In[15]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[16]:


len(X_train) , len(y_train)


# In[17]:


len(X_test) , len(y_test)


# ## ANN

# In[20]:


import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense , Activation
from tensorflow.keras.activations import relu,sigmoid,softmax


# ### Developing Architecture 

# In[23]:


X_train.shape


# In[29]:


model = Sequential()

# hidden layer 1
model.add(Dense(units=256,activation='relu',kernel_initializer='he_uniform',input_dim = X_train.shape[1]))
# hidden layer 2
model.add(Dense(units = 126 , activation='relu',kernel_initializer='he_uniform'))
# hidden layer 3
model.add(Dense(units = 63 , activation='relu',kernel_initializer='he_uniform'))
# hidden layer 4
model.add(Dense(units = 32 , activation='relu',kernel_initializer='he_uniform'))
#  hidden layer 5
model.add(Dense(units = 16 , activation='relu',kernel_initializer='he_uniform'))
# hidden layer 6
model.add(Dense(units = 8 , activation='relu',kernel_initializer='he_uniform'))
# hidden layer 7
model.add(Dense(units = 4 , activation='relu',kernel_initializer='he_uniform'))
# hidden layer 8
model.add(Dense(units = 2 , activation='relu',kernel_initializer='he_uniform'))
#  hidden layer 9
model.add(Dense(units = 1 , activation='sigmoid',kernel_initializer='glorot_uniform'))



# In[30]:


model.summary()

##  parameters information 
model = Sequential()

# hidden layer 1
model.add(Dense(units=4,activation='relu',kernel_initializer='he_uniform',input_dim = 2))
# hidden layer 2
model.add(Dense(units = 1 , activation='sigmoid',kernel_initializer='glorot_uniform'))# model.summary()
# In[31]:


model.compile(loss = 'binary_crossentropy' , optimizer = 'adam' , metrics=['accuracy'])


# In[43]:


tf.config.run_functions_eagerly(True)
model.fit(X_train,y_train , batch_size=10 , epochs=50)


# In[ ]:




