#!/usr/bin/env python
# coding: utf-8

# # Task 1- Prediction using supervised ML

# # Author - Sanyukta Khatdeo
# 
To predict the percentage of the students based on the number of hours they studied

# In[2]:


# importing the required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


#Reading data
url = "https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv"
data = pd.read_csv(url)
print("**Data Imported**")
data


# In[6]:



data.shape


# In[7]:


data.describe()


# In[8]:


# Check if there is any null value in the Dataset
data.isnull == True


# there is no null in the dataset hence we can now visualize

# # Visualization and Analysis of Dataset

# In[39]:


#plotting of distribution of scores and number of hours of study on 2D graph

data.plot(x='Hours', y='Scores', style='o')
plt.title('no. of Hours studied Vs Scores of Students')
plt.xlabel('no. of Hours studied')
plt.ylabel('Scores of students')
plt.show()


# In[20]:


#no. of hours studied = x variable
#scores = y variable
X = data.iloc[:, :-1].values
Y = data.iloc[:, 1].values


# In[21]:


#view X variable
X


# In[22]:


#view Y variable
Y


# In[ ]:





# In[25]:


#for splitting data into training and test sets

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)


# In[27]:


X_train


# In[28]:


X_test


# In[29]:


Y_train


# In[30]:


Y_test


# # Training of Machine Learning model(Alogorithm)

# In[31]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)
print("***Training of ML model is Completed***")


# # Visualizing the Model

# In[34]:


#plotting the regression line
line = regressor.coef_* X + regressor.intercept_

#plotting for the test data
plt.scatter(X,Y)
plt.plot(X, line, color ="red");
plt.show()


# # Making Predictions

# In[35]:


print(X_test)  #testing data - in hours
Y_pred = regressor.predict(X_test)  #predicting the scores


# In[36]:


#Comparing actual Vs Predicted Data
df = pd.DataFrame({'Actual':Y_test, 'Predicted':Y_pred})
df


# In[37]:


#testing with our own custom data
#score of student if he/she studies for 9.25 hrs/day

hours = 9.25
own_pred = regressor.predict([[hours]])
print("No of Hours - {}".format(hours))
print("Predicted Score - {}".format(own_pred[0]))


# # Evaluating the Model

# In[38]:


#mean absolute error:

from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, Y_pred))

#max error:
print('Max Error:', metrics.max_error(Y_test, Y_pred))

#mean squared error:
print('Mean Squared Error:', metrics.mean_squared_error(Y_test, Y_pred))


# In[ ]:




