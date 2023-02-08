#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[2]:


import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

import warnings
warnings.simplefilter("ignore")


# In[3]:


data = pd.read_csv("C:/Users/HP/Downloads/archive (7).zip")


# In[4]:


data.head(10)


# In[5]:


data["Language"].value_counts()


# In[6]:


# separating the independent and dependant features
X = data["Text"]
y = data["Language"]


# In[7]:


# converting categorical variables to numerical

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)


# ### Text Preprocessing

# In[8]:


data_list = []
for text in X:
    text = re.sub(r'[!@#$(),\n"%^*?\:;~`0-9]', ' ', text)
    text = re.sub(r'[[]]', ' ', text)
    text = text.lower()
    data_list.append(text)


# ### Bags Of Words

# In[9]:


#creating bag of words using countvectorizer

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(data_list).toarray()


# In[10]:


X.shape


# ### Train Test Split 

# In[11]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)


# ### Model Creation & Prediction 

# In[12]:


from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()
model.fit(x_train, y_train)


# In[13]:


# prediction 
y_pred = model.predict(x_test)


# ### Prediction 

# In[14]:


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

ac = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
cr = classification_report(y_test, y_pred)


# In[15]:


print("Accuracy is: ", ac)


# In[16]:


#Classification Report
print(cr)


# In[17]:


#Visulization The Confusion matrix
plt.figure(figsize = (15, 10))
sns.heatmap(cm, annot = True)
plt.show()

