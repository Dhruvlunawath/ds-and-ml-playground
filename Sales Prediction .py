#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


data = pd.read_csv("https://raw.githubusercontent.com/amankharwal/Website-data/master/advertising.csv")
print(data.head())


# In[3]:


data.shape


# In[4]:


data.info()


# In[5]:


data.isnull().sum()


# In[6]:


data.describe()


# In[7]:


import matplotlib.pyplot as plt


# In[8]:


# Assuming 'data' is your DataFrame
# Calculate the mean values of each column
mean_values = data.mean()

# Plotting the mean values as a bar plot using Pandas
mean_values.plot(kind='bar', figsize=(8, 6))  # Set the figure size if needed
plt.xlabel('Spent on')
plt.ylabel('Mean Values')
plt.title('Mean Values of Each Column')  
plt.show()


# In[9]:


import seaborn as sns
sns.heatmap(data.corr(),annot=True)


# In[10]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


# In[11]:


x=data.iloc[:,:-1]
y=data.iloc[:,-1]


# In[12]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=42)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[13]:


rf=RandomForestRegressor(n_estimators=50)
rf.fit(x_train,y_train)


# In[20]:


trainpred=rf.predict(x_train)


# In[21]:


print(trainpred)


# In[22]:


testpred=rf.predict(x_test)
print(testpred)


# In[25]:


import matplotlib.pyplot as plt


plt.plot(data.index, data['Sales'], c='k', label='Actual Sales')  
plt.plot(data.index[:len(trainpred)], trainpred, c='orange', label='Train Predictions')  
plt.plot(data.index[len(trainpred):], testpred, c='green', label='Test Predictions')  

plt.xlabel('Index or X-axis Label')  
plt.ylabel('Sales or Y-axis Label')  
plt.title('Sales Prediction Comparison')  
plt.legend()  # Show legend

plt.show()


# In[23]:


from sklearn.metrics import mean_absolute_error
print(mean_absolute_error(y_test,testpred))


# In[17]:


import numpy as np
import math
print(math.sqrt(mean_absolute_error(y_test,testpred)))


# In[18]:


a=input("Enter expected price spending on TV advertisement:")
b=input("Enter expected price spending on Radio advertisement:")
c=input("Enter expected price spending on Newspaper advertisement:")
features=np.array([[a,b,c]])
print("Expected Sales:",rf.predict(features))


# In[ ]:




