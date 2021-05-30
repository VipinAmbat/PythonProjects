#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
print('Python: {}'.format(sys.version))


# In[3]:


import scipy
print('SCIPY: {}'.format(scipy.__version__))


# In[4]:


import numpy
print('NUMPY: {}'.format(numpy.__version__))


# In[5]:


import matplotlib
print('MATPLOTLIB: {}'.format(matplotlib.__version__))


# In[6]:


import pandas
print('PANDAS: {}'.format(pandas.__version__))


# In[7]:


import sklearn
print('sklearn: {}'.format(sklearn.__version__))


# In[18]:


import pandas
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import model_selection
from sklearn.ensemble import VotingClassifier


# In[27]:


#Loading the data
import pandas as pd
from pandas import read_csv
iris_data = pd.read_csv("IRIS.csv")
names=["sepal_length","sepal_width","petal_length","petal_width","species"]


# In[35]:


#Statistsical data

iris_data.describe()


# In[43]:


#Dimension of data

iris_data.shape


# In[37]:


#take a peek of data

iris_data.head(20)


# In[42]:


#class distribution

iris_data.groupby('species').size()


# In[50]:


#create a Univariate plot -each attribute

iris_data.plot(kind='box', subplots=True, layout=(2,2) ,sharex=False ,sharey=False)
pyplot.show()



# In[46]:


iris_data.hist()
pyplot.show()


# In[51]:


#multivariate plots
# create multivariate plot- relationship between attribute

scatter_matrix(iris_data)
pyplot.show()


# In[55]:


#Creating a validation set

#splitting dataset
array=iris_data.values
X=array[:, 0:4]
Y=array[:, 4]
X_train,X_validation,Y_train,Y_validation=train_test_split(X,Y, test_size=0.2, random_state=1)


# In[62]:


#logistic regression



#Linear discriminant analysis



#K-Nearest neighbours



#gaussien naive bayes


#Support vector machine 


# Building a mode

models= []
models.append(('LR',LogisticRegression(solver='liblinear',multi_class='ovr')))
models.append(('LDA',LinearDiscriminantAnalysis()))
models.append(('KNN',KNeighborsClassifier()))
models.append(('NB',GaussianNB()))
models.append(('SVM',SVC(gamma='auto')))


# In[66]:


#evaluate the created model

results=[]

names =[]

for name,model in models:
    kfold=StratifiedKFold(n_splits=10, random_state=1)
    cv_results=cross_val_score(model,X_train,Y_train,cv=kfold,scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))


# In[67]:


#Compare the models

pyplot.boxplot(results,labels=names)
pyplot.title('Algorithm comparison')
pyplot.show()


# In[68]:


#make predcition on svm

model=SVC(gamma='auto')
model.fit(X_train,Y_train)
predictions = model.predict(X_validation)


# In[69]:


#evaluate our prediction

print(accuracy_score(Y_validation,predictions))
print(confusion_matrix(Y_validation,predictions))
print(classification_report(Y_validation,predictions))


# In[ ]:




