#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data


# In[7]:


start='2018-01-01'
end='2022-07-01'
df=data.DataReader('INFY.NS', 'yahoo', start, end)
df.head()


# In[3]:


df.tail()


# In[8]:


df = df.reset_index()
df.head()


# In[9]:


df2=df.reset_index()['Close']


# In[10]:


df2


# In[11]:


df2.shape


# In[12]:


plt.plot(df2)


# In[13]:


from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
df1=scaler.fit_transform(np.array(df2).reshape(-1,1))


# In[14]:


print(df1)


# In[15]:


##splitting dataset into train and test split
training_size=int(len(df1)*0.65)
test_size=len(df2)-training_size
train_data,test_data=df1[0:training_size,:],df1[training_size:len(df1),:1]


# In[16]:


training_size,test_size


# In[17]:


train_data


# In[18]:


def create_dataset(dataset, time_step=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-time_step-1):
		a = dataset[i:(i+time_step), 0]   
		dataX.append(a)
		dataY.append(dataset[i + time_step, 0])
	return np.array(dataX), np.array(dataY)


# In[19]:


# reshape into X=t,t+1,t+2,t+3 and Y=t+4
time_step = 100
X_train, y_train = create_dataset(train_data, time_step)
X_test, ytest = create_dataset(test_data, time_step)


# In[20]:


print(X_train.shape), print(y_train.shape)


# In[21]:


print(X_test.shape), print(ytest.shape)


# In[22]:


# reshape input to be [samples, time steps, features] which is required for LSTM
X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)


# In[23]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM


# In[24]:


model=Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(100,1)))
model.add(LSTM(50,return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')


# In[25]:


model.summary()


# In[26]:


model.summary()


# In[27]:


model.fit(X_train,y_train,validation_data=(X_test,ytest),epochs=50,batch_size=64,verbose=1)


# In[28]:


### Lets Do the prediction and check performance metrics
train_predict=model.predict(X_train)
test_predict=model.predict(X_test)


# In[29]:


##Transformback to original form
train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)


# In[30]:


### Calculate RMSE performance metrics
import math
from sklearn.metrics import mean_squared_error
math.sqrt(mean_squared_error(y_train,train_predict))


# In[45]:


from sklearn import preprocessing
import numpy as np


# In[75]:


y_train=np.random.random((1,4))
y_train=y_train*20
print("train_predict=",y_train)


# In[83]:


normalized=preprocessing.normalize(y_train)
print("train_predict=",normalized)


# In[87]:


ytest=np.random.random((1,1))
ytest=ytest*20
print("test_predict=",ytest)


# In[88]:


normalized=preprocessing.normalize(ytest)
print("test_predict=",normalized)


# In[89]:


### Plotting 
# shift train predictions for plotting
look_back=100
trainPredictPlot = np.empty_like(df1)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
# shift test predictions for plotting
testPredictPlot = np.empty_like(df1)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(df1)-1, :] = test_predict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(df1))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()


# In[90]:


len(test_data)


# In[91]:


x_input=test_data[290:].reshape(1,-1)
x_input.shape


# In[92]:


temp_input=list(x_input)
temp_input=temp_input[0].tolist()


# In[93]:


temp_input


# In[94]:


len(temp_input)


# In[95]:


# demonstrate prediction for next 10 days
from numpy import array

lst_output=[]
n_steps=100
i=0
while(i<30):
    
    if(len(temp_input)>100):
        #print(temp_input)
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        print(len(temp_input))
        lst_output.extend(yhat.tolist())
        i=i+1
    

print(lst_output)


# In[96]:


day_new=np.arange(1,101)
day_pred=np.arange(101,131)


# In[97]:


len(df1)


# In[98]:


plt.plot(day_new,scaler.inverse_transform(df1[1012:]))
plt.plot(day_pred,scaler.inverse_transform(lst_output))


# In[99]:


df3=df1.tolist()
df3.extend(lst_output)
plt.plot(df3[1000:])


# In[100]:


df3=scaler.inverse_transform(df3).tolist()


# In[101]:


plt.plot(df3)


# In[ ]:





# In[ ]:





# In[ ]:




