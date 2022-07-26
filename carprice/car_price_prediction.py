#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df= pd.read_csv('car data.csv')


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


print(df['Seller_Type'].unique())
print(df['Transmission'].unique())
print(df['Owner'].unique())


# In[ ]:





# In[6]:


df.isnull().sum()


# In[ ]:





# In[7]:


df.describe()


# In[8]:


df.columns


# In[9]:


final_dataset = df[['Year', 'Selling_Price', 'Present_Price', 'Kms_Driven',
       'Fuel_Type', 'Seller_Type', 'Transmission', 'Owner']]


# In[10]:


final_dataset['current_year'] = 2020


# In[11]:


final_dataset['no_years'] = final_dataset['current_year']-final_dataset['Year']


# In[12]:


final_dataset.drop(['Year'],axis=1,inplace=True)


# In[13]:


final_dataset.drop(['current_year'],axis=1,inplace=True)


# In[14]:


final_dataset.head()


# In[ ]:





# In[15]:


#ONE HOT ENCODING


# In[16]:


final_dataset = pd.get_dummies(final_dataset,drop_first=True)


# In[17]:


final_dataset.shape


# In[18]:


final_dataset.head()


# In[19]:


final_dataset.corr()


# In[ ]:





# In[ ]:





# In[ ]:





# In[20]:


import seaborn as sns


# In[21]:


sns.pairplot(final_dataset)


# In[22]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[23]:


corrmat = final_dataset.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
g = sns.heatmap(final_dataset[top_corr_features].corr(),annot=True,cmap='RdYlGn')


# In[ ]:





# In[ ]:





# In[24]:


#FEATURE SELECTION


# In[25]:


X = final_dataset.iloc[:,1:]
y = final_dataset.iloc[:,0]


# In[26]:


X.head()


# In[27]:


y.head()


# In[ ]:





# In[ ]:





# In[28]:


from sklearn.ensemble import ExtraTreesRegressor
model = ExtraTreesRegressor()
model.fit(X,y)


# In[29]:


print(model.feature_importances_)


# In[30]:


feat_importances = pd.Series(model.feature_importances_,index= X.columns)
feat_importances.nlargest(5).plot(kind='barh')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[31]:


#DO MACHINE LEARNING ALGORITHMS NOW


# In[32]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)


# In[ ]:





# In[33]:


from sklearn.ensemble import RandomForestRegressor
rf_random = RandomForestRegressor()


# In[ ]:





# In[34]:


#HYPERPARAMETER TUNING


# In[38]:


import numpy as np
n_estimators = [int(x) for x in np.linspace(start=100,stop=1200,num=12)]
print(n_estimators)
#number of trees in random forest

max_features = ['auto','sqrt'] # number of features in tree

max_depth = [int(x) for x in np.linspace(5,30,num=6)]  # max no of levels in tree

min_sample_split = [2,5,10,15,100] # min no of samples required to split a node

min_sample_leaf = [1,2,5,10]  # min no of samples required at each leaf node


# In[ ]:





# In[39]:


from sklearn.model_selection import RandomizedSearchCV


# In[40]:


#create random grid


# In[48]:


random_grid = {'n_estimators':n_estimators,
               'max_features':max_features,
               'max_depth':max_depth,
               'min_samples_split':min_sample_split,
               'min_samples_leaf':min_sample_leaf    
}
print(random_grid)


# In[49]:


rf = RandomForestRegressor()


# In[50]:


rf_random = RandomizedSearchCV(estimator = rf,param_distributions=random_grid,scoring='neg_mean_squared_error',n_iter=10,cv=5,verbose=2,random_state=42,n_jobs=1)


# In[51]:


rf_random.fit(X_train,y_train)


# In[ ]:





# In[52]:


#prediction


# In[53]:


predictions = rf_random.predict(X_test)
print(predictions)


# In[54]:


sns.distplot(y_test-predictions)


# In[55]:


plt.scatter(y_test,predictions)


# In[ ]:





# In[58]:


import pickle

#open a file where you wanna save a file

file = open('random_forest_regression_model.pkl','wb')

#dump info to that file

pickle.dump(rf_random,file)


# In[ ]:




