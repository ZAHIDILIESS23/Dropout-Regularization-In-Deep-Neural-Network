#!/usr/bin/env python
# coding: utf-8

# # Dropout Regularization In Deep Neural Network

# This is a dataset that describes sonar chirp returns bouncing off different services. The 60 input variables are the strength of the returns at different angles. It is a binary classification problem that requires a model to differentiate rocks from metal cylinders.

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


import warnings
warnings.filterwarnings('ignore')


# In[3]:


df = pd.read_csv("https://raw.githubusercontent.com/codebasics/deep-learning-keras-tf-tutorial/master/13_dropout_layer/sonar_dataset.csv", header=None)
df.sample(5)


# In[4]:


df.shape


# In[5]:


# check for nan values
df.isna().sum()


# In[6]:


df.columns


# In[7]:


df[60].value_counts() # label is not skewed


# In[8]:


X = df.drop(60, axis=1)
y = df[60]
y.head()


# In[9]:


y = pd.get_dummies(y, drop_first=True)
y.sample(5) # R --> 1 and M --> 0


# In[10]:



y.value_counts()


# In[11]:


X.head()


# In[12]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)


# In[13]:


X_train.head()


# # Using Deep Learning Model
# 

# ## Model without Dropout Layer

# In[14]:


import tensorflow as tf
from tensorflow import keras


# In[15]:


model = keras.Sequential([
    keras.layers.Dense(60, input_dim=60, activation='relu'),
    keras.layers.Dense(30, activation='relu'),
    keras.layers.Dense(15, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=100, batch_size=8)


# In[16]:


model.evaluate(X_test, y_test)


# Training Accuracy >>> Test Accuracy

# In[17]:


y_pred = model.predict(X_test).reshape(-1)
print(y_pred[:10])

# round the values to nearest integer ie 0 or 1
y_pred = np.round(y_pred)
print(y_pred[:10])


# In[18]:


y_test[:10]


# In[19]:


from sklearn.metrics import confusion_matrix , classification_report

print(classification_report(y_test, y_pred))


# ## Model with Dropout Layer

# In[20]:


modeld = keras.Sequential([
    keras.layers.Dense(60, input_dim=60, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(30, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(15, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1, activation='sigmoid')
])

modeld.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

modeld.fit(X_train, y_train, epochs=100, batch_size=8)


# In[21]:


modeld.evaluate(X_test, y_test)


# In[22]:


y_pred = modeld.predict(X_test).reshape(-1)
print(y_pred[:10])

# round the values to nearest integer ie 0 or 1
y_pred = np.round(y_pred)
print(y_pred[:10])


# ## You can see that by using dropout layer test accuracy increased from 0.77 to 0.81

# In[ ]:




