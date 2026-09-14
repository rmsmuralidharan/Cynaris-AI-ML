import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score

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


### cross validation techniques = validating the data only on training data and can calculate how well it performs on test data
### creating a linear regression model for calculating the performance for cross validation

model = LinearRegression(n_jobs=-1)

### kfold - cross validation = used only for regerssion 
kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

### calculating the performance of the k fold cross validation
scores = cross_val_score(
    model,
    x_train,
    y_train,
    cv=kf,
    scoring='r2'
)

print("Scores:", scores)
print("Average R²:", scores.mean())

"""
StratifiedKFold = used for classification based problems
"""