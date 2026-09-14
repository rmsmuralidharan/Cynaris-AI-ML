import pandas as pd


"""
im created much larger dataset for my upcoming model training because a samll dataset of just 120 rows are not enough to train a model and the prediction on the test data will also fail
"""


## load the new dataset

df_new = pd.read_csv('week2/data/employee_performance_cleaned_2.csv')

## validating the new dataset

print('dataset shape:', df_new.shape)
print('\nFirst 5 rows:')
print(df_new.head(5))


## columns check before merging the new dataset to the old one
print('new dataset columns:', df_new.columns.to_list())
print(len(df_new.columns))


## now validating old dataset 

df_old = pd.read_csv('week1/d3_Data_Loading_Cleaning/data/cleaned_employee_performance.csv')

## validating the old dataset
print('dataset shape:', df_old.shape)
print('\nFirst five rows of old dataset:')
print(df_old.head(5))

### columns of the old dataset
print('old dataset columns:', df_old.columns.to_list())
print(len(df_old.columns))


## now merging the two files

df_merged = pd.concat([df_old, df_new], ignore_index=True)

## validating the final dataset
print(len(df_merged))
print(df_merged.columns.to_list())

## saving the file
df_merged.to_csv('week2/data/merged.csv', index=False)
print('successfully saved our final dataset')
