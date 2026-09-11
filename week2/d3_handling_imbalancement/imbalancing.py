import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE

## loading the dataset

df = pd.read_csv('week2/d3_handling_imbalancement/data/customer_churn_imbalanced_5000.csv')

print(df.shape)
print(df.head(5))

### data validation
## 1.missing values
print('\nmissing values:')
print(df.isnull().sum())

## 2.duplicted values
print('\nduplicates:')
print(df.duplicated().sum())


### target validation
print('\nclasses:')
print(df['Churn'].value_counts())


### assiging independent and dependent features
x = df.drop(columns=['Churn'])
y = df['Churn']

print(x.head(5))
print(y.head(5))

### 1. over sampling - increasing the minorty class

over_sampling = RandomOverSampler(random_state=42)
x_resampled, y_resampled = over_sampling.fit_resample(x,y)

### before over sampling
print('\nbefore over sampling:')
print(y.value_counts())

### after over sampling
print('\nafter over sampling:')
print(y_resampled.value_counts())


### 2. under sampling - decreasing the majority class
under_sampling = RandomUnderSampler(random_state=42)

x_under_sampled, y_under_sampled = under_sampling.fit_resample(x,y)

### before under resampling
print('\nbefore under sampling:')
print(y.value_counts())

### after under sampling
print('\nafter under sampling:')
print(y_under_sampled.value_counts())  

"""
under sampling is not the best approach because it'll delete most of the records
"""

### 3. class handling using SMOTE

x = df[['Age']]
smote = SMOTE(random_state=42)

x_smote, y_smote = smote.fit_resample(x,y)

### before smote
print('\nbefore smote:')
print(y.value_counts())

### after smote
print('\nafter smote:')
print(y_smote.value_counts())

"""
difference between random over sampling and smote is random over sampling duplicates the data points untill we reach the majority class
smote synthetically creates a data points near the existing data points
"""






