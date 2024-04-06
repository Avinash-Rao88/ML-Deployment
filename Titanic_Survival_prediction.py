#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')


# ### About Data 
#  
#  
#  
# 1. survival	 (Survival	    0 = No, 1 = Yes)
# 2. pclass	 (Ticket class	1 = 1st, 2 = 2nd, 3 = 3rd)
# 3. sex	     (Sex	)
# 4. Age	     (Age in years	)
# 5. sibsp	(# of siblings / spouses aboard the Titanic	)
# 6. parch	(# of parents / children aboard the Titanic	)
# 7. ticket	(Ticket number	)
# 8. fare	    (Passenger fare	)
# 9. cabin	(Cabin number	)
# 19. embarked	(Port of Embarkation	C = Cherbourg, Q = Queenstown, S = Southampton)

# In[2]:


df=pd.read_csv('titanic.csv')
df
#df2=df.copy()


# ### Data Preprocessing

# The head() method in pandas is used to view the first few rows of a DataFrame or Series
# The tail() method in pandas is used to view the last few rows of a DataFrame or Series 
# The sample() method in pandas is used to randomly sample rows from a DataFrame or Series

# In[3]:


df
# df.head()    
# df.tail()     
# df.sample()   


# # Description of basic methods
# 
# 
# 
# 1. The shape attribute in pandas is used to determine the dimensions (number of rows and columns) of a DataFrame or a
# NumPy array.
# 
# 2. The info() method in pandas is used to retrieve a concise summary of the DataFrame's information, including data types, 
# non-null values, and memory usage
# 
# 3. The describe() method in pandas is used to generate descriptive statistics of a DataFrame or Series.
# 4. The isnull() method in pandas is used to check for missing or null values in a DataFrame or Series.
# 5. The nunique() method in pandas is used to count the number of unique values in a Series (a single column of a DataFrame)
# 6. The unique() method in pandas is used to retrieve the unique values from a Series (a single column of a DataFrame)
# 7. The value_counts() method in pandas is used to count the frequency of each unique value in a Series (a single column of    a DataFrame).
# 8. By default, dropna() removes any row that contains at least one null value.
# 9. The fillna() method in pandas is used to fill missing or null (NaN) values in a DataFrame or Series with specified    values or using various filling strategies.

# In[4]:


df.shape


# In[5]:


df.info()


# In[6]:


df.describe()


# In[7]:


df['Pclass'].unique()


# In[8]:


df['Pclass'].nunique()


# In[9]:


df['Embarked'].value_counts()


# ### What to do if your dataset has null values?
# 1. You can drop all the rows that has null value
# 2. You can fill the null values with mean,median, mode.

# In[10]:


df.isnull().sum().sort_values(ascending=False)


# In[11]:


df['Age']=df['Age'].fillna(df['Age'].mean())
df.isnull().sum()


# In[12]:


(df['Cabin'].isnull().sum()/df.shape[0])*100


# In[13]:


# df=df.dropna()
df=df.dropna(subset=['Embarked'])   #it will drop row that contains null value in Embarked column.


# In[14]:


df.isnull().sum()


# # Visualisation

# In[15]:


#histogram is used to plot continuous data
plt.hist(df['Age'])
plt.plot()


# In[16]:


import seaborn as sns
sns.countplot(df,x='Embarked')


# In[17]:


import seaborn as sns
sns.countplot(df,x='Sex')


# A Label Encoder is a preprocessing technique used in machine learning and data analysis to convert categorical data into numerical form. Categorical data represents categories or labels, such as colors, cities, or types of fruits, and these labels need to be converted into numerical values for many machine learning algorithms to work effectively, as most algorithms require numerical input.
# 
# One-Hot Encoding is a popular technique for representing categorical data in a format that is suitable for machine learning algorithms. It is used to convert categorical variables into a binary matrix (1s and 0s), where each category is transformed into a new column, and each column represents a category with a binary value indicating its presence or absence.

# In[18]:


from sklearn.preprocessing import LabelEncoder
lr=LabelEncoder()
df['Sex']=lr.fit_transform(df['Sex'])
df['Embarked']=lr.fit_transform(df['Embarked'])


# In[19]:


df.head()


# In[20]:


#The drop() method in pandas is used to remove specified rows or columns from a DataFrame.
ndf=df.drop(['Name','PassengerId','Ticket','Cabin'],axis=1)
ndf


# ### Correlation Matrix
# A correlation matrix is a table or matrix that displays the correlation coefficients between many variables. It is a common tool in statistics and data analysis to understand the relationships between variables, especially in multivariate datasets. Correlation coefficients quantify the strength and direction of a linear relationship between two variables, and they typically range from -1 to 1.

# In[21]:


corrM=ndf.corr()
corrM


# In[22]:


import seaborn as sns
sns.heatmap(corrM,cmap='PiYG',annot=True)
plt.show()


# ### Outlier
# An outlier is an observation or data point that significantly deviates from the majority of other data points in a dataset. In other words, it is a data point that lies far outside the typical range of values in a dataset. Outliers can occur in various types of data, including numerical and categorical data, and they can have a significant impact on statistical analyses and machine learning models.

# In[23]:


#Outlier Detection using boxplot
sns.boxplot(ndf)


# In[24]:


sns.boxplot(ndf['Fare'])


# In[25]:


x=df['Fare']
y=df['Age']
plt.scatter(x,y)
plt.xlabel("Fare")
plt.ylabel("Age")
plt.grid()
plt.title("Scatter Plot")
plt.plot()


# ### Outlier Removal

# In[26]:


ndf.describe()


# In[27]:


percentile25 = ndf['Fare'].quantile(0.25)
percentile75 = ndf['Fare'].quantile(0.75)


# In[28]:


percentile25


# In[29]:


percentile75


# In[30]:


iqr=percentile75-percentile25  #IQR- INter Quantile Range


# In[31]:


upper_limit = percentile75 + 1.5 * iqr
lower_limit = percentile25 - 1.5 * iqr


# In[32]:


lower_limit,upper_limit


# In[33]:


new_df = ndf[(ndf['Fare'] <= 100) & (ndf['Fare'] >= 0)]


# In[34]:


new_df


# In[35]:


x=new_df['Fare']
y=new_df['Age']
plt.scatter(x,y)
plt.xlabel("Fare")
plt.ylabel("Age")
plt.grid()
plt.title("Scatter Plot")
plt.plot()


# In[36]:


sns.boxplot(new_df['Fare'])


# In[37]:


new_df


# In[38]:


corrM=new_df.corr()


# In[39]:


sns.heatmap(corrM,annot=True,cmap='PiYG')


# In[40]:


x=new_df.drop(['Survived'],axis=1)
y=new_df['Survived']


# In[41]:


x


# In[42]:


y


# In[43]:


from sklearn.model_selection import train_test_split


# In[44]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=21)


# In[45]:


x_train.shape,y_train.shape,x_test.shape,y_test.shape


# In[46]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,classification_report


# ### RandomForestClassifier

# In[47]:


rf=RandomForestClassifier()


# In[48]:


rf.fit(x_train,y_train)


# In[49]:


print("Training Accuracy")
print(rf.score(x_train,y_train))
print("Testing Accuracy")
print(rf.score(x_test,y_test))


# In[50]:


pred=rf.predict(x_test)
pred[:5]


# In[51]:


y_test[:5]


# In[52]:


cm=confusion_matrix(pred,y_test)


# In[53]:


sns.heatmap(cm,annot=True)


# In[54]:


print(classification_report(pred,y_test))


# ### Hyperparameter Tuning

# In[55]:


from sklearn.model_selection import GridSearchCV ,RandomizedSearchCV


# #### Hyperparameter Tuning on Random Forest Classifier 
# 
# 1. max_features{“sqrt”, “log2”, None}, int or float, default=”sqrt”
# The number of features to consider when looking for the best split:
# If “sqrt”, then max_features=sqrt(n_features).
# If “log2”, then max_features=log2(n_features).
# If None, then max_features=n_features.
# 
# 2. max_leaf_nodes, int, default=None
# Grow trees with max_leaf_nodes in best-first fashion. Best nodes are defined as relative reduction in impurity. If None then unlimited number of leaf nodes.
# 
# 3. max_depth int, default=None
# The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.
# 
# 4. n_estimators int, default=100
# The number of trees in the forest.

# In[56]:


param_grid = {
    'n_estimators': [25, 50, 100, 150],
    'max_features': ['sqrt', 'log2', None],
    'max_depth': [3, 6, 9],
    'max_leaf_nodes': [3, 6, 9],
}


# In[57]:


rf_cv = RandomizedSearchCV(rf, param_grid, cv = 5)
# rf_cv = GridSearchCV(rf, param_grid, cv = 5)


# In[58]:


rf_cv.fit(x_train,y_train)


# In[59]:


rf_cv.score(x_train,y_train),rf_cv.score(x_test,y_test)


# In[60]:


rf_cv.best_params_


# # Save the Model

# In[62]:


import pickle
with open('trained_model.pkl', 'wb') as model_file:
    pickle.dump(rf_cv, model_file)


# In[ ]:





# In[ ]:





# In[ ]:




