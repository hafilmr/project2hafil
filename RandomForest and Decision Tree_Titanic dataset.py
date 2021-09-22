#!/usr/bin/env python
# coding: utf-8

# In[46]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.model_selection import cross_val_score, KFold
import warnings
warnings.filterwarnings('ignore')


# In[48]:


df_train = pd.read_csv("train_clean.csv")
df_test = pd.read_csv("test_clean.csv")

df_train.describe(include='all')


# In[4]:


print(df_train.head(3))
print(df_test.head(3))
print(df_train.shape)
print(df_test.shape)


# In[5]:


print(df_train.info())


# In[6]:


print(df_test.info())


# In[7]:


print("Missing values pada df_train : ")
print(df_train.isna().sum())
print("Missing values pada df_test : ")
print(df_test.isna().sum())


# In[8]:


print(df_train["Embarked"].value_counts())


# In[9]:


df_train["Age"].fillna(df_train["Age"].median(), inplace = True)
df_test["Age"].fillna(df_test["Age"].median(), inplace = True)
df_train["Embarked"].fillna('S', inplace = True)


# In[10]:


df_train.hist(bins = 11, figsize = (18,10))
plt.show()


# In[11]:


korelasi = df_train[["Survived", "SibSp", "Parch", "Age", "Pclass", "Fare"]].corr()
sns.heatmap(korelasi, annot = True)
plt.show()


# In[50]:


sns.barplot(x="Sex", y="Survived", data=df_train)

#print percentages of females vs. males that survive
print("Percentage of females who survived:", df_train["Survived"][df_train["Sex"] == 'female'].value_counts(normalize = True)[1]*100)

print("Percentage of males who survived:", df_train["Survived"][df_train["Sex"] == 'male'].value_counts(normalize = True)[1]*100)


# In[51]:


sns.barplot(x="Pclass", y="Survived", data=df_train)

#print percentage of people by Pclass that survived
print("Percentage of Pclass = 1 who survived:", df_train["Survived"][df_train["Pclass"] == 1].value_counts(normalize = True)[1]*100)

print("Percentage of Pclass = 2 who survived:", df_train["Survived"][df_train["Pclass"] == 2].value_counts(normalize = True)[1]*100)

print("Percentage of Pclass = 3 who survived:", df_train["Survived"][df_train["Pclass"] == 3].value_counts(normalize = True)[1]*100)


# In[52]:


sns.barplot(x="SibSp", y="Survived", data=df_train)

#I won't be printing individual percent values for all of these.
print("Percentage of SibSp = 0 who survived:", df_train["Survived"][df_train["SibSp"] == 0].value_counts(normalize = True)[1]*100)

print("Percentage of SibSp = 1 who survived:", df_train["Survived"][df_train["SibSp"] == 1].value_counts(normalize = True)[1]*100)

print("Percentage of SibSp = 2 who survived:", df_train["Survived"][df_train["SibSp"] == 2].value_counts(normalize = True)[1]*100)


# In[53]:


sns.barplot(x="Parch", y="Survived", data=df_train)
plt.show()


# In[54]:


df_train["Age"] = df_train["Age"].fillna(-0.5)
df_test["Age"] = df_test["Age"].fillna(-0.5)
bins = [-1, 0, 5, 12, 18, 24, 35, 60, np.inf]
labels = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
df_train['AgeGroup'] = pd.cut(df_train["Age"], bins, labels = labels)
df_test['AgeGroup'] = pd.cut(df_test["Age"], bins, labels = labels)

#draw a bar plot of Age vs. survival
sns.barplot(x="AgeGroup", y="Survived", data=df_train)
plt.show()


# In[55]:


df_train["CabinBool"] = (df_train["Cabin"].notnull().astype('int'))
df_test["CabinBool"] = (df_test["Cabin"].notnull().astype('int'))

#calculate percentages of CabinBool vs. survived
print("Percentage of CabinBool = 1 who survived:", df_train["Survived"][df_train["CabinBool"] == 1].value_counts(normalize = True)[1]*100)

print("Percentage of CabinBool = 0 who survived:", df_train["Survived"][df_train["CabinBool"] == 0].value_counts(normalize = True)[1]*100)
#draw a bar plot of CabinBool vs. survival
sns.barplot(x="CabinBool", y="Survived", data=df_train)
plt.show()


# In[56]:


df_test.describe(include='all')


# In[57]:


df_train = df_train.drop(['Cabin'], axis = 1)
df_test = df_test.drop(['Cabin'], axis = 1)


# In[58]:


df_train = df_train.drop(['Ticket'], axis = 1)
df_test = df_test.drop(['Ticket'], axis = 1)


# In[59]:


print("Number of people embarking in Southampton (S):")
southampton = df_train[df_train["Embarked"] == "S"].shape[0]
print(southampton)

print("Number of people embarking in Cherbourg (C):")
cherbourg = df_train[df_train["Embarked"] == "C"].shape[0]
print(cherbourg)

print("Number of people embarking in Queenstown (Q):")
queenstown = df_train[df_train["Embarked"] == "Q"].shape[0]
print(queenstown)


# In[60]:


df_train = df_train.fillna({"Embarked": "S"})


# In[61]:


combine = [df_train, df_test]

#extract a title for each Name in the train and test datasets
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(df_train['Title'], df_train['Sex'])


# In[62]:


for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Capt', 'Col',
    'Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Rare')
    
    dataset['Title'] = dataset['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

df_train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()


# In[63]:


title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Royal": 5, "Rare": 6}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

df_train.head()


# In[64]:


mr_age = df_train[df_train["Title"] == 1]["AgeGroup"].mode() #Young Adult
miss_age = df_train[df_train["Title"] == 2]["AgeGroup"].mode() #Student
mrs_age = df_train[df_train["Title"] == 3]["AgeGroup"].mode() #Adult
master_age = df_train[df_train["Title"] == 4]["AgeGroup"].mode() #Baby
royal_age = df_train[df_train["Title"] == 5]["AgeGroup"].mode() #Adult
rare_age = df_train[df_train["Title"] == 6]["AgeGroup"].mode() #Adult

age_title_mapping = {1: "Young Adult", 2: "Student", 3: "Adult", 4: "Baby", 5: "Adult", 6: "Adult"}

#I tried to get this code to work with using .map(), but couldn't.
#I've put down a less elegant, temporary solution for now.
#train = train.fillna({"Age": train["Title"].map(age_title_mapping)})
#test = test.fillna({"Age": test["Title"].map(age_title_mapping)})

for x in range(len(df_train["AgeGroup"])):
    if df_train["AgeGroup"][x] == "Unknown":
        df_train["AgeGroup"][x] = age_title_mapping[train["Title"][x]]
        
for x in range(len(df_test["AgeGroup"])):
    if df_test["AgeGroup"][x] == "Unknown":
        df_test["AgeGroup"][x] = age_title_mapping[test["Title"][x]]


# In[66]:


age_mapping = {'Baby': 1, 'Child': 2, 'Teenager': 3, 'Student': 4, 'Young Adult': 5, 'Adult': 6, 'Senior': 7}
df_train['AgeGroup'] = df_train['AgeGroup'].map(age_mapping)
df_test['AgeGroup'] = df_test['AgeGroup'].map(age_mapping)

df_train.head()

#dropping the Age feature for now, might change
df_train = df_train.drop(['Age'], axis = 1)
df_test = df_test.drop(['Age'], axis = 1)


# In[67]:


df_train = df_train.drop(['Name'], axis = 1)
df_test = df_test.drop(['Name'], axis = 1)


# In[68]:


sex_mapping = {"male": 0, "female": 1}
df_train['Sex'] = df_train['Sex'].map(sex_mapping)
df_test['Sex'] = df_test['Sex'].map(sex_mapping)

df_train.head()


# In[69]:


embarked_mapping = {"S": 1, "C": 2, "Q": 3}
df_train['Embarked'] = df_train['Embarked'].map(embarked_mapping)
df_test['Embarked'] = df_test['Embarked'].map(embarked_mapping)

df_train.head()


# In[70]:


for x in range(len(df_test["Fare"])):
    if pd.isnull(df_test["Fare"][x]):
        pclass = df_test["Pclass"][x] #Pclass = 3
        df_test["Fare"][x] = round(df_train[df_train["Pclass"] == pclass]["Fare"].mean(), 4)
        
#map Fare values into groups of numerical values
df_train['FareBand'] = pd.qcut(df_train['Fare'], 4, labels = [1, 2, 3, 4])
df_test['FareBand'] = pd.qcut(df_test['Fare'], 4, labels = [1, 2, 3, 4])

#drop Fare values
df_train = df_train.drop(['Fare'], axis = 1)
df_test = df_test.drop(['Fare'], axis = 1)


# In[71]:


df_train.head()


# In[72]:


df_test.head()


# In[73]:


from sklearn.model_selection import train_test_split

predictors = df_train.drop(['Survived', 'PassengerId'], axis=1)
target = df_train["Survived"]
x_train, x_val, y_train, y_val = train_test_split(predictors, target, test_size = 0.22, random_state = 0)


# In[74]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

decisiontree = DecisionTreeClassifier()
decisiontree.fit(x_train, y_train)
y_pred = decisiontree.predict(x_val)
acc_decisiontree = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_decisiontree)


# In[75]:


from sklearn.ensemble import RandomForestClassifier

randomforest = RandomForestClassifier()
randomforest.fit(x_train, y_train)
y_pred = randomforest.predict(x_val)
acc_randomforest = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_randomforest)


# In[ ]:




