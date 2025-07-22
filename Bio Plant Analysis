#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# In[2]:


data = pd.read_csv('C:/Users/Laksh/Downloads/Bio Plant Data.csv')


# In[3]:


data.info()


# In[4]:


data = data.dropna()
data = data.drop("East Basin Phenol", axis=1)
data = data.drop("West Basin Phenol (ppm)", axis=1)
data = data.drop("Final Clarifier Phenol (ppm)", axis=1)


# In[5]:


col_names = data.columns.tolist()


# In[6]:


primary = []
for i in col_names:
    if "Primary" in i:
        primary.append(data[i])
primary = pd.DataFrame(primary).T
x_primary = primary.drop("Primary pH" , axis=1)
y_primary = primary["Primary pH"]


East_Basin = []
for i in col_names:
    if "East Basin" in i:
        East_Basin.append(data[i])
East_Basin = pd.DataFrame(East_Basin).T
x_East_Basin = East_Basin.drop("East Basin pH" , axis=1)
y_East_Basin = East_Basin["East Basin pH"]

West_Basin = []
for i in col_names:
    if "West Basin" in i:
        West_Basin.append(data[i])
West_Basin = pd.DataFrame(West_Basin).T
x_West_Basin = West_Basin.drop("West Basin pH" , axis=1)
y_West_Basin = West_Basin["West Basin pH"]


Final_Clarifier = []
for i in col_names:
    if "Final Clarifier" in i:
        Final_Clarifier.append(data[i])
Final_Clarifier = pd.DataFrame(Final_Clarifier).T
x_Final_Clarifier = Final_Clarifier.drop("Final Clarifier pH" , axis=1)
y_Final_Clarifier = Final_Clarifier["Final Clarifier pH"]


# In[7]:


def pre_processing(X,y):
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def linear_regression(X,y):
    X_train, X_test, y_train, y_test = pre_processing(X,y)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return mse

def neural_network(X,y,activation_function):
    X_train, X_test, y_train, y_test = pre_processing(X,y)
    model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_dim=X_train.shape[1]),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32, activation=activation_function),
    tf.keras.layers.Dense(1)
])
    model.compile(optimizer='adam', loss='mean_squared_error')
    history = model.fit(X_train, y_train, epochs=500, batch_size=len(y_train), validation_data=(X_test, y_test))
    
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    plt.plot(history.history['loss'], label='train loss')
    plt.plot(history.history['val_loss'], label='validation loss')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    return mse
    


# In[8]:


primary_dict = {}
primary_dict.update({"tanh":neural_network(x_primary,y_primary,"tanh")})
primary_dict.update({"sigmoid":neural_network(x_primary,y_primary,"sigmoid")})
primary_dict.update({"relu":neural_network(x_primary,y_primary,"relu")})
primary_dict.update({"linear regression":linear_regression(x_primary,y_primary)})


# In[9]:


East_Basin_dict = {}
East_Basin_dict.update({"tanh":neural_network(x_East_Basin,y_East_Basin,"tanh")})
East_Basin_dict.update({"sigmoid":neural_network(x_East_Basin,y_East_Basin,"sigmoid")})
East_Basin_dict.update({"relu":neural_network(x_East_Basin,y_East_Basin,"relu")})
East_Basin_dict.update({"linear regression":linear_regression(x_East_Basin,y_East_Basin)})


# In[10]:


West_Basin_dict = {}
West_Basin_dict.update({"tanh":neural_network(x_West_Basin,y_West_Basin,"tanh")})
West_Basin_dict.update({"sigmoid":neural_network(x_West_Basin,y_West_Basin,"sigmoid")})
West_Basin_dict.update({"relu":neural_network(x_West_Basin,y_West_Basin,"relu")})
West_Basin_dict.update({"linear regression":linear_regression(x_West_Basin,y_West_Basin)})


# In[11]:


Final_Clarifier_dict = {}
Final_Clarifier_dict.update({"tanh":neural_network(x_Final_Clarifier,y_Final_Clarifier,"tanh")})
Final_Clarifier_dict.update({"sigmoid":neural_network(x_Final_Clarifier,y_Final_Clarifier,"sigmoid")})
Final_Clarifier_dict.update({"relu":neural_network(x_Final_Clarifier,y_Final_Clarifier,"relu")})
Final_Clarifier_dict.update({"linear regression":linear_regression(x_Final_Clarifier,y_Final_Clarifier)})


# In[12]:


print("Primary:", primary_dict)
print("East Basin",East_Basin_dict)
print("West_Basin",West_Basin_dict)
print("Final_Clarifier",Final_Clarifier_dict)

