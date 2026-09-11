import pandas as pd

### loading the titanic dataset
df = pd.read_csv('week2/d5_pipeline/data/sythetic_titanic_dataset.csv')

print(df.shape)
print(df.head(5))

## data inspection

## 1.missing values
print('\nmissing values:')
print(df.isnull().sum())

## 2.duplicate values
print('\nduplicate values:')
print(df.duplicated().sum())

### 3. datatypes
print('\ndata types:')
print(df.dtypes)

"""
age = 8020 missing values
cabin 34883 missing values
250 duplicate values
"""

print(df['Cabin'].value_counts())

### data cleaning

###1. filling the missing values
df['Age'] = df['Age'].fillna(df['Age'].mode()[0])
print('\nage after filling missing values:')
print(df['Age'].isnull().sum())


df['Cabin'] = df['Cabin'].fillna('Unknown')
print('\nCabin after filling missing values:')
print(df['Cabin'].isnull().sum())


### 2. removing the duplicate values
print('\n duplicate rows:')
print(df['PassengerId'].duplicated())

df = df.drop_duplicates()
print('\nafter removing duplicated values:')
print(df.duplicated().sum())

### final validation

## 1.missing values
print('\nmissing values:')
print(df.isnull().sum())

## 2.duplicate values
print('\nduplicate values:')
print(df.duplicated().sum())

