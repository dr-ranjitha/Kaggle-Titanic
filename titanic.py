# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 17:19:43 2020

@author: drran
"""
# Importing the libraries
# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

# Importing the dataset
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
combine = [train, test]

#Analyze by describing data
#Which features are available in the dataset?
print(train.columns.values)

# preview the data
#print('_'*40)

# Review survived rate using `percentiles=[.61, .62]` knowing our problem description mentions 38% survival rate.
# Review Parch distribution using `percentiles=[.75, .8]`
# SibSp distribution `[.68, .69]`
# Age and Fare `[.1, .2, .3, .4, .5, .6, .7, .8, .9, .99]`

#1. Exploratory Data Analysis (EDA) with Visualization
#Go through the dataset iteratively to study the features
#Get some idea about the training data
train.head()
train.info()
train.shape
train.describe()
train.describe(include=['O'])#describe(include = ['O']) will show the descriptive statistics of object data types.
train.isnull().sum()
train.tail()
#Get some idea about the testing data
test.head()
test.info()
test.shape
test.describe()
test.describe(include=['O'])#describe(include = ['O']) will show the descriptive statistics of object data types.
test.isnull().sum() #missing values

#Relationship between Features and Survival
# How many people survived?
plt.figure(figsize=(20,1))
sns.countplot(y='Survived', data=train);
print(train.Survived.value_counts())

#Pclass
train.Pclass.isnull().sum()
train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
sns.distplot(train.Pclass)
sns.countplot(y='Pclass', data=train)
train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
plot_count_dist(train, 
                bin_df=train, 
                label_column='Survived', 
                target_column='Pclass', 
                figsize=(20, 10))

grid = sns.FacetGrid(train, col='Survived', row='Pclass', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend();

#Sex
train.Sex.isnull().sum()
plt.figure(figsize=(20, 5))
sns.countplot(y="Sex", data=train)

# plt.figure(figsize=(10, 10))
# sns.distplot(train.loc[train['Survived'] == 1]['Sex'], kde_kws={'label': 'Survived'})
# sns.distplot(train.loc[train['Survived'] == 0]['Sex'], kde_kws={'label': 'Did not survive'})

train[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)

#Feature: Age
train.Age.isnull().sum()

g = sns.FacetGrid(train, col='Survived')
g.map(plt.hist, 'Age', bins=20)

#Feature: SibSp
# How many missing values does SibSp have?
train.SibSp.isnull().sum()
# What values are there?
train.SibSp.value_counts()
plt.figure(figsize=(20, 5))
sns.countplot(y="SibSp", data=train)

plot_count_dist(train, 
                bin_df=train, 
                label_column='Survived', 
                target_column='SibSp', 
                figsize=(20, 10))

train[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)

#Feature: Parch
# How many missing values does Parch have?
train.Parch.isnull().sum()

# What values are there?
train.Parch.value_counts()
# Visualise the counts of Parch and the distribution of the values
# against Survived
plot_count_dist(train, 
                bin_df=train,
                label_column='Survived', 
                target_column='Parch', 
                figsize=(20, 10))

train[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)

#Feature: Ticket
train.Ticket.isnull().sum()
# What values are there?
train.Ticket.value_counts()
#How many kinds of ticket are there?
sns.countplot(y="Ticket", data=train)

#Feature: Fare
train.Fare.isnull().sum()
# What values are there?
train.Fare.value_counts()
#How many kinds of ticket are there?
sns.countplot(y="Fare", data=train)

train.Fare.dtype
# How many unique kinds of Fare are there?
print("There are {} unique Fare values.".format(len(train.Fare.unique())))

#Feature: Cabin
#Description: The cabin number where the passenger was staying.

# How many missing values does Cabin have?
train.Cabin.isnull().sum()

# What do the Cabin values look like?
train.Cabin.value_counts()

#Feature: Embarked
#Description: The port where the passenger boarded the Titanic.
#Key: C = Cherbourg, Q = Queenstown, S = Southampton

# How many missing values does Embarked have?
train.Embarked.isnull().sum()

# What kind of values are in Embarked?
train.Embarked.value_counts()

grid = sns.FacetGrid(train, row='Embarked', size=2.2, aspect=1.6)
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
grid.add_legend()

grid = sns.FacetGrid(train, row='Embarked', col='Survived', size=2.2, aspect=1.6)
grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
grid.add_legend()

#Wrangle data
print("Before", train.shape, test.shape, combine[0].shape, combine[1].shape)

train_df = train.drop(['Ticket', 'Cabin'], axis=1)
test_df = test.drop(['Ticket', 'Cabin'], axis=1)
combine = [train_df, test_df]

"After", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape

#Extract titles from names to fill missing ages
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(train_df['Title'], train_df['Sex'])


for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()

#convert the categorical titles to ordinal
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

train_df.head()

#Drop Name feature


#Function Definitions
def plot_count_dist(data, bin_df, label_column, target_column, figsize=(20, 5), use_bin_df=False):
    """
    Function to plot counts and distributions of a label variable and 
    target variable side by side.
    ::param_data:: = target dataframe
    ::param_bin_df:: = binned dataframe for countplot
    ::param_label_column:: = binary labelled column
    ::param_target_column:: = column you want to view counts and distributions
    ::param_figsize:: = size of figure (width, height)
    ::param_use_bin_df:: = whether or not to use the bin_df, default False
    """
    if use_bin_df: 
        fig = plt.figure(figsize=figsize)
        plt.subplot(1, 2, 1)
        sns.countplot(y=target_column, data=bin_df);
        plt.subplot(1, 2, 2)
        sns.distplot(data.loc[data[label_column] == 1][target_column], 
                     kde_kws={"label": "Survived"});
        sns.distplot(data.loc[data[label_column] == 0][target_column], 
                     kde_kws={"label": "Did not survive"});
    else:
        fig = plt.figure(figsize=figsize)
        plt.subplot(1, 2, 1)
        sns.countplot(y=target_column, data=data);
        plt.subplot(1, 2, 2)
        sns.distplot(data.loc[data[label_column] == 1][target_column], 
                     kde_kws={"label": "Survived"});
        sns.distplot(data.loc[data[label_column] == 0][target_column], 
                     kde_kws={"label": "Did not survive"});