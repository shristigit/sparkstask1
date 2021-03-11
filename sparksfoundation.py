#!/usr/bin/env python
# coding: utf-8

# # SARKS FOUNDATION TASK 1
# 

# In[15]:


import pandas as pd
import numpy as np
import seaborn as sb


# In[17]:


student=pd.read_csv('http://bit.ly/w-data')


# In[24]:


student.head(20)


# In[11]:


student.describe()


# In[25]:


student.info()


# In[33]:


sb.scatterplot(x='Hours',y='Scores',data=student)


# In[34]:


import statsmodels.api as sm


# In[22]:


x=sm.add_constant(student['hours'])


# In[23]:


y=sm.OLS(student['scores'],x).fit()


# In[24]:


y.summary()


# In[39]:


from sklearn.linear_model import LinearRegression


# In[40]:


stu_score=LinearRegression()


# In[41]:


x=student[['Hours']]
y=student['Scores']


# In[42]:


stu_score.fit(x,y)


# In[55]:


stu_score.predict(x)


# In[57]:


print(stu_score.intercept_)


# In[58]:


print(stu_score.coef_)


# In[64]:


sb.jointplot(x=student['Hours'],y=['Scores'],data=student,kind='reg')


# In[10]:


stu_score.predict([[9.25]])


# In[ ]:




