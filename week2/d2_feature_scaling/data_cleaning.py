import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder

### loading the mental-health of the employees dataset
df = pd.read_csv('week2/d2_feature_scaling/data/mental_health_dataset.csv')

print(df.shape)
print(df.head(5))


## data validation
## 1.checking missing values
print('\nMissing values:')
print(df.isnull().sum())   ### no missing values found

## 2.duplicates
print('\nDuplicates')
print(df.duplicated().sum())

## 3. data type inspection
print('\ndata types of each columns:')
print(df.dtypes)

### 4. unique value counts in string columns
str_columns = [
    'employment_status',
    'work_environment',
    'mental_health_history',
    'seeks_treatment',
    'mental_health_risk'
]

for feature in str_columns:
    print(f"\n{feature}:")
    print(df[feature].value_counts())



