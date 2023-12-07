#!/usr/bin/env python
# coding: utf-8

# In[189]:


import pandas as pd
import numpy as np


# In[190]:


df=pd.read_csv(r'https://raw.githubusercontent.com/amankharwal/Website-data/master/onlinefoods.csv')
df


# In[191]:


df.drop('Output',axis=1,inplace=True)


# In[192]:


df['Reorder']=df['Unnamed: 12']


# In[193]:


df.drop('Unnamed: 12',axis=1,inplace=True)


# In[194]:


df.isnull().sum()


# In[195]:


df.info()


# In[196]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[197]:


buying_again_data = df.query("Reorder == 'Yes'")
print(buying_again_data.head())


# In[198]:


buying_again_data.shape


# In[199]:


sns.countplot(x='Gender',data=buying_again_data)
buying_again_data['Gender'].value_counts()
plt.title('Who Orders Food Online More: Male Vs. Female')
plt.show()


# In[200]:


sns.countplot(x='Marital Status',data=buying_again_data)
buying_again_data['Marital Status'].value_counts()
plt.title('Who Orders Food Online More based on Marital Status ')
plt.show()


# In[201]:


sns.countplot(x='Occupation',data=buying_again_data)
buying_again_data['Occupation'].value_counts()
plt.title('Who Orders Food Online More based on Occupation ')
plt.show()


# In[202]:


sns.countplot(x='Monthly Income',data=buying_again_data)
buying_again_data['Monthly Income'].value_counts()
plt.title('Who Orders Food Online More based on Monthly Income ')
plt.xticks(rotation=90)
plt.show()


# In[203]:


sns.countplot(x='Educational Qualifications',data=buying_again_data)
buying_again_data['Educational Qualifications'].value_counts()
plt.title('Who Orders Food Online More based on Educational Qualifications ')
plt.xticks(rotation=90)
plt.show()


# In[204]:


sns.countplot(x='Family size',data=buying_again_data)
buying_again_data['Family size'].value_counts()
plt.title('Who Orders Food Online More based on Family size ')
plt.show()


# In[205]:


sns.countplot(x='Feedback',data=buying_again_data)
buying_again_data['Feedback'].value_counts()
plt.title('Feedback of people who ordered food ')
plt.show()


# In[206]:


df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})
df["Marital Status"] = df["Marital Status"].map({"Married": 2, 
                                                     "Single": 1, 
                                                     "Prefer not to say": 0})
df["Occupation"] = df["Occupation"].map({"Student": 1, 
                                             "Employee": 2, 
                                             "Self Employeed": 3, 
                                             "House wife": 4})
df["Educational Qualifications"] = df["Educational Qualifications"].map({"Graduate": 1, 
                                                                             "Post Graduate": 2, 
                                                                             "Ph.D": 3, "School": 4, 
                                                                             "Uneducated": 5})
df["Monthly Income"] = df["Monthly Income"].map({"No Income": 0, 
                                                     "25001 to 50000": 37500, 
                                                     "More than 50000":75000, 
                                                     "10001 to 25000": 17500, 
                                                     "Below Rs.10000": 5000})
df["Feedback"] = df["Feedback"].map({"Positive": 1, "Negative ": 0})
df["Reorder"] = df["Reorder"].map({"Yes": 1, "No": 0})


# In[207]:


df.drop(df[['latitude','longitude']],axis=1,inplace=True)
df.head()


# In[208]:


from sklearn.model_selection import train_test_split


# In[209]:


x=df.iloc[:,:-1].values
y=df.iloc[:,-1].values


# In[210]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


# In[211]:


print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


# In[212]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


# In[213]:


rf = RandomForestClassifier()
rf.fit(x_train, y_train)
rf_pred=rf.predict(x_test)
rf_pred


# In[214]:


lr =LogisticRegression()
lr.fit(x_train, y_train)
lr_pred=lr.predict(x_test)
lr_pred


# In[215]:


from sklearn.metrics import accuracy_score
print('Random Forest Classifier:',accuracy_score(rf_pred,y_test))
print('Logistic Regression:',accuracy_score(lr_pred,y_test))


# In[216]:


avg=(lr_pred+rf_pred)//2
accuracy_score(avg,y_test)


# In[219]:


print("Enter Customer Details to Predict If the Customer Will Order Again")
a = int(input("Enter the Age of the Customer: "))
b = int(input("Enter the Gender of the Customer (1 = Male, 0 = Female): "))
c = int(input("Marital Status of the Customer (1 = Single, 2 = Married, 3 = Not Revealed): "))
d = int(input("Occupation of the Customer (Student = 1, Employee = 2, Self Employeed = 3, House wife = 4): "))
e = int(input("Monthly Income: "))
f = int(input("Educational Qualification (Graduate = 1, Post Graduate = 2, Ph.D = 3, School = 4, Uneducated = 5): "))
g = int(input("Family Size: "))
h = int(input("Pin Code: "))
i = int(input("Review of the Last Order (1 = Positive, 0 = Negative): "))
features = np.array([[a, b, c, d, e, f, g, h, i]])
print("Finding if the customer will order again: ", rf.predict(features))

