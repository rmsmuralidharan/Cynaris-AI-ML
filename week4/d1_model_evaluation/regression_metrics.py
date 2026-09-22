from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd

data =  fetch_california_housing(as_frame = True)

df = data.frame

print(df.shape)
print(df.head(2))
print(df.info())


### missing values
print('\nmissing values:')
print(df.isnull().sum())

### duplicate values
print('\nduplicate values:')
print(df.duplicated().sum())



## dependent and independent features
x = df.drop(columns = ['MedHouseVal'])
y = df['MedHouseVal']


### train test split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

## scaling
scaler = StandardScaler()

x_train_scaled = scaler.fit_transform(x_train)

x_test_scaled = scaler.transform(x_test)

# ----------------------
### linear regression
# -----------------------

linear_regression = LinearRegression()

## model training

linear_regression.fit(x_train_scaled, y_train)

## model prediction

y_pred_linear = linear_regression.predict(x_test_scaled)

#------------------
## ridge regression
# ------------------

ridge = Ridge(alpha=0.1)

ridge.fit(x_train_scaled, y_train)

y_pred_ridge = ridge.predict(x_test_scaled)

#------------------
## lasso regression
#------------------

lasso = Lasso(alpha=0.01)

lasso.fit(x_train_scaled, y_train)

y_pred_lasso = lasso.predict(x_test_scaled)


#-----------------------
## decision tree 
# -----------------------

dt = DecisionTreeRegressor(max_depth=5,random_state=42)

dt.fit(x_train, y_train)

y_pred_dt = dt.predict(x_test)


#----------------
##random forest
#----------------

rf = RandomForestRegressor(n_estimators=100,max_depth=5,random_state=42)

rf.fit(x_train, y_train)

y_pred_rf = rf.predict(x_test)


#------------------
##gradient boosting
#------------------

gradient = GradientBoostingRegressor(
    n_estimators=100,
    random_state=42
)

gradient.fit(x_train_scaled,y_train)

ypred_gd = gradient.predict(x_test_scaled)



# ------------------------------------------------------
# Model Evaluation
# ------------------------------------------------------

models = {
    "Linear Regression": (linear_regression, x_test_scaled),
    "Ridge Regression": (ridge, x_test_scaled),
    "Lasso Regression": (lasso, x_test_scaled),
    "Decision Tree": (dt, x_test),
    "Random Forest": (rf, x_test),
    "Gradient Boosting": (gradient, x_test_scaled)
}

results = []

for name, (model, x_eval) in models.items():

    # Prediction
    y_pred = model.predict(x_eval)

    # Metrics
    mae = mean_absolute_error(y_test, y_pred)

    mse = mean_squared_error(y_test, y_pred)

    rmse = np.sqrt(mse)

    r2 = r2_score(y_test, y_pred)

    # Adjusted R²
    n = x_eval.shape[0]
    p = x_eval.shape[1]

    adjusted_r2 = 1 - (
        (1 - r2) * (n - 1) / (n - p - 1)
    )

    results.append({
        "Model": name,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R²": r2,
        "Adjusted R²": adjusted_r2
    })


# ------------------------------------------------------
# Create Results Table
# ------------------------------------------------------

regression_results = pd.DataFrame(results)

# Round values
regression_results = regression_results.round(4)

print("\nRegression Model Evaluation")
print("=" * 90)

print(regression_results.to_string(index=False))