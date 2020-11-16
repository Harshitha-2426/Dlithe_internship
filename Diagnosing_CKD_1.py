#!/usr/bin/env python
# coding: utf-8

# # Diagnosing Chronic Kidney Disease (CKD) with 100% Accuracy

# # Description

# Chronic kidney disease (CKD), also known as chronic renal disease. Chronic kidney disease involves
# conditions that damage your kidneys and decrease their ability to keep you healthy. You may develop complications like high
# blood pressure, anemia (low blood count), weak bones, poor nutritional health and nerve damage. 
# 
# Attribute Information:
# Matrix column entries (attributes): 
# -->The age of person-age,
# -->blood pressure level-bp,
# -->Sugar level-su,
# -->Specific gravity-sg,
# -->Red blood cells-rgb,
# -->pus cell-pc,
# -->pus cell clumps-pcc,
# -->Blood gulcose random-bgr,
# -->Blood Urea-Bu,
# -->Bacteria-Ba,
# -->Serum Creatinine-sc,
# -->Sodium-sod,
# -->Potassium-pot,
# -->Hemoglobin-hemo,
# -->Packed cell volume-pcv,
# -->White Blood cell count-wbcc,
# -->Red Blood cell count-rbcc,
# -->Hypertension-htn,
# -->Diabetes Mellitus-dm,
# -->Coronary Artery Disease-cad,
# -->Appetite-appet,
# -->Pedal Edema-pe,
# -->Anemia-ane,
# -->classes-Target

# ## Summary
# 
# In this project, I use Logistic Regression and K-Nearest Neighbors (KNN) to diagnose CKD. Both were able to classify patients with 100% accuracy.
# 
# KNN required class balancing, scaling, and model tuning to perform with 100% accuracy, while Logistic Regression was 100% accurate without tuning (note: still had to stratify the train test split). 
# 
# Logistic Regression is deemed a better model for this case, because in addition to being 100% accurate, it also allows us to quantify the impact of unit increases in specific variables on likelihood of having CKD.

# # data collection

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import scipy.stats as stats
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#reading the input file and storing in data
df = pd.read_csv('C:/Users/Public/Desktop (2)/chronic_kidney_disease.csv')
data = df


# # data interpetation

# In[3]:


#printing the data i.e, the input dataset
data


# In[4]:


#printing the data attributes with datatype
data.info()
type(data)


# In[5]:


#displaying the first 5 values
data.head()


# In[6]:


#displaying the last 5 values
data.tail()


# In[7]:


data.describe()


# # Data Mapping

# In[8]:


data['class'] = data['class'].map({'ckd':1,'notckd':0})
data['htn'] = data['htn'].map({'yes':1,'no':0})
data['dm'] = data['dm'].map({'yes':1,'no':0})
data['cad'] = data['cad'].map({'yes':1,'no':0})
data['appet'] = data['appet'].map({'good':1,'poor':0})
data['ane'] = data['ane'].map({'yes':1,'no':0})
data['pe'] = data['pe'].map({'yes':1,'no':0})
data['ba'] = data['ba'].map({'present':1,'notpresent':0})
data['pcc'] = data['pcc'].map({'present':1,'notpresent':0})
data['pc'] = data['pc'].map({'abnormal':1,'normal':0})
data['rbc'] = data['rbc'].map({'abnormal':1,'normal':0})


# In[9]:


#checking the count of class-TARGET
data['class'].value_counts()


# Factors that may increase your risk of chronic kidney disease include:
# 
# - Diabetes - su(blood sugar), dm (diabetes mellitus)
# - High blood pressure - BP
# - Heart and blood vessel (cardiovascular) disease
# - Smoking
# - Obesity
# - Being African-American, Native American or Asian-American
# - Family history of kidney disease
# - Abnormal kidney structure
# - Older age - age

# # EXPLANATORY DATA ANALYSIS

# In[10]:


#Figuring the plot with help of heatmap
plt.figure(figsize = (19,19))
sns.heatmap(data.corr(), annot = True, cmap = 'coolwarm') # looking for strong correlations with "class" row 


# # Explortary data analysis-EDA

# In[37]:


#Checking the size of input datset
data.shape


# In[12]:


#displaying the columns in dataset
data.columns


# In[13]:


#checking if NULL values are present
data.isnull().sum()


# In[14]:


#Checking the size of input datset
data.shape[0], data.dropna().shape[0]


# We would only have 158 rows remaining if we drop the na columns.  One downside is that we reduce the overall power of our model when we feed in less data, and another is that we dont know if the fact that those values are null is related in some way to an additional variable.  If the latter is the case, throwing out that data could potentially skew our data.
# 
# I am going to drop in this case and see how the model performs.
# 

# In[15]:


data.dropna(inplace=True)


# In[16]:


#Checking the shape of the data
data.shape


# # Modeling

# ## Logistic Regression

# In[17]:


from sklearn.linear_model import LogisticRegression


# In[18]:


logreg = LogisticRegression()


# In[19]:


# splitting the concept and target 
X = data.iloc[:,:-1]
y = data['class']


# In[20]:


#DATA ANALYSIS METHOD-VISUALIZATION
#To know the count  of values in the target we plot a count graph#0=notckd#1=ckd
sns.countplot(data['class'])


# In[21]:


#split the input data for testing and training
X_train, X_test, y_train, y_test = train_test_split(X,y, stratify = y, shuffle = True)


# In[22]:


#Fitting the data into x and y train
logreg.fit(X_train,y_train)


# In[23]:


#Prediction of the X test and train
test_pred = logreg.predict(X_test)
train_pred = logreg.predict(X_train)


# In[24]:


from sklearn.metrics import accuracy_score, confusion_matrix


# In[25]:


#Printing the train and test accuracy
print('Train Accuracy: ', accuracy_score(y_train, train_pred))
print('Test Accuracy: ', accuracy_score(y_test, test_pred))


# ### The cell below shows the coefficients for each variable.
# (example on reading the coefficients from a Logistic Regression: a one unit increase in age makes an individual about e^0.14 time as likely to have ckd, while a one unit increase in blood pressure makes an individual about e^-0.07 times as likely to have ckd.

# In[26]:


#printing dataframe of columns
pd.DataFrame(logreg.coef_, columns=X.columns)


# In[27]:


#Prediting test prediction of y test
tn, fp, fn, tp = confusion_matrix(y_test, test_pred).ravel()

print(f'True Neg: {tn}')
print(f'False Pos: {fp}')
print(f'False Neg: {fn}')
print(f'True Pos: {tp}')


# ## K-Nearest Neighbors Classifier

# I am going to balance the classes here before using KNN. Logistic regression was able to make accurate predictions even when trained on unbalanced classes, but KNN is more sensitive to unbalanced classes

# In[38]:


#defining the counts in class
df["class"].value_counts()


# In[39]:


#resting index value
balanced_df = pd.concat([df[df["class"] == 0], df[df["class"] == 1].sample(n = 115, replace = True)], axis = 0)
balanced_df.reset_index(drop=True, inplace=True)


# In[41]:


#printing class count after resting
balanced_df["class"].value_counts()


# In[31]:


X = balanced_df.drop("class", axis=1)
y = balanced_df["class"]


# In[32]:


X_train, X_test, y_train, y_test = train_test_split(X,y, random_state = 42)


# In[42]:


#Fitting the data and tranforming it
ss = StandardScaler()
ss.fit(X_train)
X_train = ss.transform(X_train)
X_test = ss.transform(X_test)


# In[43]:


from sklearn.neighbors import KNeighborsClassifier


# In[44]:


#Predicting the accuracy 
knn = KNeighborsClassifier()
params = {
    "n_neighbors":[3,5,7,9],
    "weights":["uniform","distance"],
    "algorithm":["ball_tree","kd_tree","brute"],
    "leaf_size":[25,30,35],
    "p":[1,2]
}
gs = GridSearchCV(knn, param_grid=params)
model = gs.fit(X_train,y_train)
preds = model.predict(X_test)
accuracy_score(y_test, preds)


# In[55]:


#confusion_matrix prediction
tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()

print(f'True Neg: {tn}')
print(f'False Pos: {fp}')
print(f'False Neg: {fn}')
print(f'True Pos: {tp}')


# In[ ]:




