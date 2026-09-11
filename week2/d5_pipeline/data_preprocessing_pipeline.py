import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

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


### removing the unwanted features
df = df.drop(columns=['PassengerId', 'Name', 'Ticket'])
print('\nfeatures after removing:')
print(df.head(2))


### Data preprocessing

## 1. ordinal encoding for the sex feature
ordinal_encoder = OrdinalEncoder(
    categories=[['female', 'male']]
)

ordinal_encoded = ordinal_encoder.fit_transform(df[['Sex']])

ordinal_df = pd.DataFrame(
    ordinal_encoded
)

print('\nordinal encoding:')
for category, value in enumerate(ordinal_encoder.categories_[0]):
    print(f"{category} ---> {value}")



### 2. one hot encoding for embarked and cabin

onehot_encoder = OneHotEncoder(
    sparse_output=False,
    handle_unknown='ignore'
)

onehot_encoded = onehot_encoder.fit_transform(df[['Embarked', 'Cabin']])

onehot_df = pd.DataFrame(
    onehot_encoded,
    columns=onehot_encoder.get_feature_names_out()
)

print(onehot_df.head(5))

### merging the final preprocessed dataset
df_final = pd.concat([df, ordinal_df, onehot_df], axis=1)

df_final = df_final.drop(columns=['Embarked', 'Cabin'])

### final preprocessed dataset

print('\nbefore preprocessing the dataset:')
print(df.shape)

print('\nafter preprocessing the dataset:')
print(df_final.shape)

