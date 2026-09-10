import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

### load the employee dataset
df = pd.read_csv('week2/data/merged.csv')

print('\nshape:')
print(df.shape)

print('\nfirst 5 rows:')
print(df.head(5))

### checking the datatypes
print('\ndata types:')
print(df.dtypes)

### one hot encoding for the department column

encoder = OneHotEncoder(
    sparse_output=False,
    handle_unknown='ignore'
)

encoded = encoder.fit_transform(df[['Department']])

encoded_df = pd.DataFrame(
    encoded,
    columns=encoder.get_feature_names_out()
)

print(encoded_df.head(5))

### remove the department column
df = df.drop(columns=['Department', 'Employee_ID'])

### merging the encoded columns with the existing columns

df_merged = pd.concat([df, encoded_df], axis=1)
print(df_merged.head(5))


### assiging depemdemt and independent features
x = df_merged.drop(columns=['Performance_Score'])
y = df_merged['Performance_Score']


### train test split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=42)

### displaying their shapes
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(x_test.shape)


