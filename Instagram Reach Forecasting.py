#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df=pd.read_csv(r"C:\Users\dhruv\Downloads\Instagram-Reach.csv")


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


df['Date'] = pd.to_datetime(df['Date'])
print(df.head())


# In[6]:


df.shape


# In[7]:


import matplotlib.pyplot as plt

x = df['Date']
y = df['Instagram reach']

# Plotting the data
plt.plot(x, y)
plt.xlabel('Date')
plt.ylabel('Instagram reach')
plt.title('Instagram Reach Over Time')
plt.show()


# In[8]:


df['Day'] = df['Date'].dt.day_name()
print(df.head())


# In[9]:


import numpy as np

day_stats = df.groupby('Day')['Instagram reach'].agg(['mean', 'median', 'std']).reset_index()
print(day_stats)


# In[10]:


import matplotlib.pyplot as plt

# Assuming 'day_stats' is a DataFrame with columns 'Day' and 'mean'

# Create a line plot
plt.bar(day_stats['Day'], day_stats['mean'], label='Mean')

# Add labels and title
plt.xlabel('Day')
plt.ylabel('Mean Value')
plt.title('Mean Value for Each Day')
plt.xticks(rotation=45)


# Show the plot
plt.show()


# In[11]:


import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

df = df[["Date", "Instagram reach"]]

result = seasonal_decompose(df['Instagram reach'], 
                            model='multiplicative', 
                            period=100)

fig = plt.figure()
fig = result.plot()
fig.show()


# In[12]:


pd.plotting.autocorrelation_plot(df["Instagram reach"])


# In[13]:


from statsmodels.graphics.tsaplots import plot_pacf
plot_pacf(df["Instagram reach"], lags = 100)


# In[14]:


p, d, q = 8, 1, 2

import statsmodels.api as sm
import warnings
model=sm.tsa.statespace.SARIMAX(df['Instagram reach'],
                                order=(p, d, q),
                                seasonal_order=(p, d, q, 12))
model=model.fit()
print(model.summary())


# In[18]:


predictions = model.predict(len(df),len(df)+100)


# In[19]:


predictions


# In[24]:


import matplotlib.pyplot as plt

# Plot training data
plt.plot(df.index, df["Instagram reach"], label="Training Data", linestyle="-")

# Plot predictions
plt.plot(predictions.index, predictions, label="Predictions", linestyle="-")

# Add labels and legend
plt.xlabel("Index")
plt.ylabel("Reach")
plt.title("Reach forecasting")
plt.legend()

# Show the plot
plt.show()

