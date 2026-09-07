import pandas as pd

## load the dataset
df = pd.read_csv('week1/d3_Data_Loading_Cleaning/data/employee_performance.csv')

### basic inspection

print('\ndataset shape:')
print(df.shape)


print('\nColumns:')
print(df.columns.tolist())

print('\ndata types:')
print(df.dtypes)

print('\nFirst 10 rows')
print(df.head(10))

print('\nMissing values:')
print(df.isnull().sum())

print('\nduplicate rows:')
print(df.duplicated().sum())


## data cleaning
## remove duplicated rows

print('\nshape before removing the duplicates:')
print(df.shape)

print('\nshape after removing duplicates:')

df = df.drop_duplicates()
print(df.shape)


## handling missing values

## fill missing values age with median
df['Age'] = df['Age'].fillna(df['Age'].median())

## fill  missing salary with median
df['Salary'] = df['Salary'].fillna(df['Salary'].median())

## fill missing performance sccore with median
df['Performance_Score'] = df['Performance_Score'].fillna(df['Performance_Score'].median())


print('\nMissing values after cleaning:')
print(df.isnull().sum())


## final inspection

print('\nfinal dataset shape:')
print(df.shape)

print('\nfinal missing values:')
print(df.isnull().sum())

print('\nfinal duplicate count:')
print(df.duplicated().sum())

## save the cleaned dataset
df.to_csv('week1/d3_Data_Loading_Cleaning/data/cleaned_employee_performance.csv', index=False)

print('\ncleaned dataset saved successfully.')

