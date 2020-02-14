#!/usr/bin/env python
# coding: utf-8

# Hello, world! Welcome to my first complete Data Science project.  This project will cover most of the steps in a common Data Science framework:
# 
# 1. Problem: What are you trying to solve? 
# 
# 2. Gather the data: There is a big chance that the data is somewhere out there and you need to find it.
# 
# 3. Prepare the data: After data collection, we need to process our data (outliers, missing data etc.) in order to create more accurate models.
# 
# 4. Data exploration: Visualise and explore our data to identify patterns, variable types, categorisation and formulate hypothesis for our research. Furthermore, this step is likely to help us identify the models we need to look for.
# 
# 5. Data modelling: Similar to data exploration, this step will give us some insight in our data. Additionally, it can predict future outcomes and determine the algorithms we can use to improve our results. 
# 
# 6. Model validation and implementation: After training the model on a small subset of the entire dataset, it is the time to test the model on the entire dataset to check the accuracy and avoid overfitting/underfitting of the model. 
# 
# 7. Model optimisation: This step requires you to go back to your model and check what can be improved. It is mostly done on a trial and error basis.
# 

# # 1. What am I trying to solve?
# 
# This project is trying to predict whether a transaction recorded in the dataset is fraudulent or not based on an existing dataset from Université Libre de Bruxelles. 
# 

# # 2. Data gathering
# 
# This step is already done for us by research collaboration of Worldline and the Machine Learning Group (http://mlg.ulb.ac.be) of ULB (Université Libre de Bruxelles) on big data mining and fraud detection. More details on current and past projects on related topics are available on https://www.researchgate.net/project/Fraud-detection-5 and the page of the DefeatFraud project.

# # 3. Data exploration 
# 
# This project did not require data gathering, hence I avoided data architecture, governance, and extraction steps. We can only focus on data cleaning and preparation for modelling. 
# 
# The datasets contains transactions made by credit cards in September 2013 by european cardholders. This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.
# 
# It contains only numerical input variables which are the result of a PCA transformation. Unfortunately, due to confidentiality issues, the authors cannot provide the original features and more background information about the data. Features V1, V2, ... V28 are the principal components obtained with PCA, the only features which have not been transformed with PCA are 'Time' and 'Amount'. Feature 'Time' contains the seconds elapsed between each transaction and the first transaction in the dataset. The feature 'Amount' is the transaction Amount, this feature can be used for example-dependant cost-senstive learning. Feature 'Class' is the response variable and it takes value 1 in case of fraud and 0 otherwise.
# 
# Since our features are already PCA transformed by the authors, we can assume that they scaled the variables before doing the PCA Dimensionality Reduction. (Variables V1 through V28 are obtained with PCA). 
# 
# ### Load all packages required for the project

# In[1]:


#load packages
import sys #access to system parameters https://docs.python.org/3/library/sys.html
print("Python version: {}". format(sys.version))

import pandas as pd #collection of functions for data processing and analysis modeled after R dataframes with SQL like features
print("pandas version: {}". format(pd.__version__))

import matplotlib #collection of functions for scientific and publication-ready visualization
print("matplotlib version: {}". format(matplotlib.__version__))

import numpy as np #foundational package for scientific computing
print("NumPy version: {}". format(np.__version__))

import scipy as sp #collection of functions for scientific computing and advance mathematics
print("SciPy version: {}". format(sp.__version__)) 

import IPython
from IPython import display #pretty printing of dataframes in Jupyter notebook
print("IPython version: {}". format(IPython.__version__)) 

import sklearn #collection of machine learning algorithms
print("scikit-learn version: {}". format(sklearn.__version__))


#misc libraries
import collections
import time


#ignore warnings
import warnings
warnings.filterwarnings('ignore')
print('-'*45)


# ### Data Modelling algorithms, visualisation libraries and data preparation tools

# In[2]:



#Visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns

#Configure Visualization Defaults
#%matplotlib inline = show plots in Jupyter Notebook browser
get_ipython().run_line_magic('matplotlib', 'inline')
mpl.style.use('ggplot')
sns.set_style('white')
pylab.rcParams['figure.figsize'] = 12,8


# In[21]:


data = pd.read_csv("~/Downloads/creditcard.csv") 


# In[4]:


#check the first 5 rows
data.head()


# In[5]:


#check summary statistics
data.describe()


# In[6]:


#Check for NAs in the dataset - this dataset does not contain missing values
data.isnull().sum()


# In[7]:


data.columns


# In[8]:


f,ax=plt.subplots(15,2,figsize=(12,60))
#f.delaxes(ax)
col = list(data)
col = [e for e in col if e not in ('Class')]

# Let's generate distplots for our data
for i,feature in enumerate(col):
    sns.distplot(data[data['Class']==1].dropna()[(feature)], ax=ax[i//2,i%2], kde_kws={"color":"black"}, hist=False )
    sns.distplot(data[data['Class']==0].dropna()[(feature)], ax=ax[i//2,i%2], kde_kws={"color":"black"}, hist=False )

    # Get the two lines from the ax[i//2,i%2]es to generate shading
    l1 = ax[i//2,i%2].lines[0]
    l2 = ax[i//2,i%2].lines[1]

    # Get the xy data from the lines so that we can shade
    x1 = l1.get_xydata()[:,0]
    y1 = l1.get_xydata()[:,1]
    x2 = l2.get_xydata()[:,0]
    y2 = l2.get_xydata()[:,1]
    ax[i//2,i%2].fill_between(x2,y2, color="deeppink", alpha=0.6)
    ax[i//2,i%2].fill_between(x1,y1, color="darkturquoise", alpha=0.6)

    #grid
    ax[i//2,i%2].grid(b=True, which='major', color='grey', linewidth=0.3)
    
    ax[i//2,i%2].set_title('{} by target'.format(feature), fontsize=18)
    ax[i//2,i%2].set_ylabel('count', fontsize=12)
    ax[i//2,i%2].set_xlabel('Modality', fontsize=12)

    #sns.despine(ax[i//2,i%2]=ax[i//2,i%2], left=True)
    ax[i//2,i%2].set_ylabel("frequency", fontsize=12)
    ax[i//2,i%2].set_xlabel(str(feature), fontsize=12)

plt.tight_layout()
plt.show()


# In[9]:


#Check the outcome variable ['Class']

class_counter = pd.value_counts(data['Class'], sort = True).sort_index()
class_counter.plot(kind = 'bar')
plt.title("Outcome variable count")
plt.xlabel("0 = Not Fraudulent, 1 = Fraudulent")
plt.ylabel("Count")

#We can see that we have 284315 instances of 0 and only 492 fraudulent transactions
#which suggests that our data is highly skewed which we need to fix


# In[10]:


#Plotting distribution of ['Amount', 'Time'] variables in the dataset

fig, ax = plt.subplots(1, 2, figsize=(18,4))

amount_val = data['Amount'].values
time_val = data['Time'].values

sns.distplot(amount_val, ax=ax[0], color='green')
ax[0].set_title('Distribution of Transaction Amount', fontsize=14)
ax[0].set_xlim([min(amount_val), max(amount_val)])

sns.distplot(time_val, ax=ax[1], color='magenta')
ax[1].set_title('Distribution of Transaction Time', fontsize=14)
ax[1].set_xlim([min(time_val), max(time_val)])

plt.show()


# ## Scaling and distribution
# 
# Take a moment to look at the graphs plotted above. Outcome variable Class is highly skewed because most of the transactions are non-fraudulent. Similarly, Amount and Time variables are skewed because of the nature of the data, and unlike other variables V1 through V28 which are products of principal component analysis, Amount and Time are not scaled.   
# 
# 

# In[11]:


# Let's scale Amount and Time
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler

#Robust scaler is better for data with a lot of outliers

standard_scaler = StandardScaler()
robust_scaler = RobustScaler()

data['AmountScaled'] = robust_scaler.fit_transform(data['Amount'].values.reshape(-1,1))
data['TimeScaled'] = robust_scaler.fit_transform(data['Time'].values.reshape(-1,1))

data.drop(['Amount', 'Time'], axis = 1, inplace = True)


# In[12]:


amount_scaled = data['AmountScaled']
time_scaled = data['TimeScaled']
data.drop(['AmountScaled', 'TimeScaled'], axis = 1, inplace = True)
data.insert(0, 'amount_scaled', amount_scaled)
data.insert(1, 'time_scaled', time_scaled)
data.head()


# ## Splitting the dataset with random undersampling 
# 
# We split the dataset because in a sub-sample because our data is heavily skewed. Our outcome variable Class assumes that only a tiny fraction of cases is fraudulent. Hence, we might get biased results and we want our model to actually identify the potential fraud rather than implicitly assuming that it is not.
# 
# We will now proceed to create a 50/50 ratio in the dataset.

# In[13]:


X = data.ix[:, data.columns != 'Class']
y = data.ix[:, data.columns == 'Class']

# Number of data points in the minority class
number_records_fraud = len(data[data.Class == 1])
fraud_indices = np.array(data[data.Class == 1].index)

# Picking the indices of the normal classes
normal_indices = data[data.Class == 0].index

# Out of the indices we picked, randomly select "x" number (number_records_fraud)
random_normal_indices = np.random.choice(normal_indices, number_records_fraud, replace = False)
random_normal_indices = np.array(random_normal_indices)

# Appending the 2 indices
under_sample_indices = np.concatenate([fraud_indices,random_normal_indices])

# Under sample dataset
under_sample_data = data.iloc[under_sample_indices,:]

X_undersample = under_sample_data.ix[:, under_sample_data.columns != 'Class']
y_undersample = under_sample_data.ix[:, under_sample_data.columns == 'Class']

# Showing ratio
print("Percentage of normal transactions: ", len(under_sample_data[under_sample_data.Class == 0])/len(under_sample_data))
print("Percentage of fraud transactions: ", len(under_sample_data[under_sample_data.Class == 1])/len(under_sample_data))
print("Total number of transactions in resampled data: ", len(under_sample_data))


# In[14]:


# Let's investigate correlation of our new subsample

f, (ax1, ax2) = plt.subplots(2, 1, figsize =(24,20))

corr = data.corr()
sns.heatmap(corr, cmap='rainbow', annot_kws={'size':20}, ax=ax1)
ax1.set_title("Imbalanced Correlation Matrix \n (Original Dataset with scaled features Time and Amount)", fontsize=16)

sub_sample_corr = under_sample_data.corr()
sns.heatmap(sub_sample_corr, cmap='rainbow', annot_kws={'size':20}, ax=ax2)
ax2.set_title('SubSample Correlation Matrix', fontsize=16)
plt.show()


# ## Logistic Regression

# In[15]:


X = X_undersample
y = y_undersample
from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#Turn variables into arrays in order to apply the Logistic Regression
X_train = X_train.values
X_test = X_test.values
y_train = y_train.values
y_test = y_test.values


# In[16]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
classifier = LogisticRegression()

classifier.fit(X_train, y_train)
training_score = cross_val_score(classifier, X_train, y_train, cv = 5)
print('Logistic regression has a training score of: ', round(training_score.mean(), 2), 'or ',round(training_score.mean(), 2)*100, '%')


# ### GridSearchCV to find the most suitable parameters for classifier (in this case Logistic Regression)

# In[17]:


# Use GridSearchCV to find the best parameters.
from sklearn.model_selection import GridSearchCV


# Logistic Regression 
log_reg_params = {"penalty": ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}



grid_log_reg = GridSearchCV(LogisticRegression(), log_reg_params)
grid_log_reg.fit(X_train, y_train)
# We automatically get the logistic regression with the best parameters.
log_reg = grid_log_reg.best_estimator_
log_reg.fit(X_train, y_train)

log_reg_score = cross_val_score(log_reg, X_train, y_train, cv=5)
print('Logistic Regression Cross Validation Score: ', round(log_reg_score.mean() * 100, 2).astype(str) + '%')


# ### Confusion Matrix

# ![1*Z54JgbS4DUwWSknhDCvNTQ.png](attachment:1*Z54JgbS4DUwWSknhDCvNTQ.png)
# Photo source: Understanding Confusion Matrix (Sarang Narkhede for TowardsDataScience.com)

# In[18]:


from sklearn.metrics import confusion_matrix

y_pred = log_reg.predict(X_test)

log_reg_cf = confusion_matrix(y_test, y_pred)
conf_matrix_plt = sns.heatmap(log_reg_cf, annot = True)
conf_matrix_plt.set_title("Logistic Regression \n Confusion Matrix", fontsize=14)
conf_matrix_plt.set_xticklabels(['', ''], fontsize=18, rotation=90)
conf_matrix_plt.set_yticklabels(['', ''], fontsize=18, rotation=360)

plt.show()


# ### Classification Report 

# In[19]:


from sklearn.metrics import classification_report

classification_rep = classification_report(y_test, y_pred)
print('Logistic Regression Model Classification Report: ', classification_rep)

