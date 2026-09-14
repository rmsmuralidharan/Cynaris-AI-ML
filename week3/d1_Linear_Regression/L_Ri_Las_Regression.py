import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import root_mean_squared_error, mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

### loaing the dataset that i used in week 2 = employee performance score
df = pd.read_csv('week2/data/merged.csv')

print(df.head(5))
print(df.shape)


### data validation
print('\nmissing values:')
print(df.isnull().sum())


print('\nduplicate values:')
print(df.duplicated().sum())

print('\ndata types:')
print(df.dtypes)

### since validation is perfect there is no need of data cleaning = we'll move directly to data preprocessing
### data preprocessing

print('\ndistinct value counts:')
print(df['Department'].value_counts())


### one hot encoder for deparrtment column
onehot_encoder = OneHotEncoder(
    sparse_output=False,
    handle_unknown='ignore'
)

onehot_encoded = onehot_encoder.fit_transform(df[['Department']])

onehot_df = pd.DataFrame(
    onehot_encoded,
    columns=onehot_encoder.get_feature_names_out()
)

print('\none hot encoded:')
print(onehot_df.head(5))


### adding the encoded column
df_new = pd.concat([df, onehot_df], axis=1)

print(df_new.head(5))


### independent and dependent feature
x = df_new.drop(columns=['Employee_ID', 'Salary', 'Department'])

y = df_new['Salary']

print(x.head(2))
print(y.head(2))




### train test split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.10, random_state=42)

### normalization only for training data
scaler = StandardScaler()

x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

### model training = 1. Linear Regression
l_regression = LinearRegression()

### 2.Ridge Regression
ridge = Ridge(alpha=0.1)

ridge.fit(x_train_scaled, y_train)

### 3.Lasso Regression
lasso = Lasso(alpha=0.1)

lasso.fit(x_train_scaled, y_train)

l_regression.fit(x_train_scaled, y_train)

### prediction 
y_pred_lr = l_regression.predict(x_test_scaled)
y_pred_ridge = ridge.predict(x_test_scaled)
y_pred_lasso = lasso.predict(x_test_scaled)


### printing slopes and intercept
print("\nSlopes and intercept:")

print("\nLinear Regression:")
print("Slopes:", l_regression.coef_)
print("Intercept:", l_regression.intercept_)

print("\nRidge Regression:")
print("Slopes:", ridge.coef_)
print("Intercept:", ridge.intercept_)

print("\nLasso Regression:")
print("Slopes:", lasso.coef_)
print("Intercept:", lasso.intercept_)



### model evaluation = 1. Linear Regression

mse = mean_squared_error(y_test, y_pred_lr)
rmse = root_mean_squared_error(y_test, y_pred_lr)
mae = mean_absolute_error(y_test, y_pred_lr)

lr_score = r2_score(y_test, y_pred_lr)

print('\nLinear regression model evaluation:')
print(
    f"mse: {mse}\n"
    f"rmse: {rmse}\n"
    f"mae: {mae}\n"
    f"lr_score: {lr_score}"
)
### model evaluation = 2. Ridge Regression

mse_ridge = mean_squared_error(y_test, y_pred_ridge)
rmse_ridge = root_mean_squared_error(y_test, y_pred_ridge)
mae_ridge = mean_absolute_error(y_test, y_pred_ridge)

ridge_score = r2_score(y_test, y_pred_ridge)

print('\nRidge regression model evaluation:')
print(
    f"mse: {mse_ridge}\n"
    f"rmse: {rmse_ridge}\n"
    f"mae: {mae_ridge}\n"
    f"ridge_score: {ridge_score}"
)

### model evaluation = 3. Lasso Regression

mse_lasso = mean_squared_error(y_test, y_pred_lasso)
rmse_lasso = root_mean_squared_error(y_test, y_pred_lasso)
mae_lasso = mean_absolute_error(y_test, y_pred_lasso)

lasso_score = r2_score(y_test, y_pred_lasso)

print('\nLasso regression model evaluation:')
print(
    f"mse: {mse_lasso}\n"
    f"rmse: {rmse_lasso}\n"
    f"mae: {mae_lasso}\n"
    f"lasso_score: {lasso_score}"
)











