#!/usr/bin/env python
# coding: utf-8

# In[14]:


import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


# In[15]:


car_predict=pd.read_csv('Cars.csv') 
car_predict uhbubu


# In[16]:


car_predict.dtypes


# In[17]:


car_predict.shape


# In[18]:


car_predict.isna().sum()


# In[19]:


car_predict.describe()


# ## Assumptiom Check

# In[22]:


sns.lmplot( x='HP', y='MPG',data=car_predict)
plt.show()


# In[23]:


sns.lmplot(x='VOL',y='MPG',data=car_predict)
plt.show()


# In[25]:


sns.lmplot( x='SP',y='MPG',data=car_predict)
plt.show()


# In[26]:


sns.lmplot(x='WT',y='MPG',data=car_predict)
plt.show()


# ## Linearity failed

# ## Normality test

# In[29]:


sns.kdeplot( data=car_predict,x='HP')
plt.show()


# In[31]:


sns.kdeplot( data=car_predict,x='VOL')
plt.show()


# In[32]:


sns.kdeplot( data=car_predict,x='SP')
plt.show()


# In[33]:


sns.kdeplot( data=car_predict,x='WT')
plt.show()


# ## Model buliding

# In[51]:


x=car_predict[['HP','VOL','SP','WT']]
y = car_predict[['MPG']]                


# ## Model Training

# In[52]:


from sklearn.linear_model import LinearRegression


# In[53]:


linear_model=LinearRegression()


# In[54]:


linear_model.fit(x,y)


# In[55]:


linear_model.coef_


# In[56]:


linear_model.intercept_


# ## Model Testing

# In[57]:


y


# In[58]:


y_pred=linear_model.predict(x)
y_pred


# ## Model Evaluation

# In[60]:


error = y - y_pred
error


# ## Homoscedasticity Check

# In[61]:


x


# In[ ]:

print("hello mlr")



# %%
