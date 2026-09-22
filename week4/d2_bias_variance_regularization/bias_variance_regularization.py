# ============================================================
# BIAS-VARIANCE TRADEOFF & REGULARIZATION
# House Price Prediction using California Housing Dataset
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing

from sklearn.model_selection import (
    train_test_split,
    KFold,
    cross_val_score
)

from sklearn.preprocessing import (
    StandardScaler,
    PolynomialFeatures
)

from sklearn.pipeline import Pipeline

from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet
)

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)


# ============================================================
# 1. LOAD DATASET
# ============================================================

print("=" * 70)
print("1. LOADING CALIFORNIA HOUSING DATASET")
print("=" * 70)

data = fetch_california_housing(as_frame=True)

X = data.data
y = data.target

print("\nDataset Shape:")
print(X.shape)

print("\nFeatures:")
print(X.columns.tolist())

print("\nFirst 5 Rows:")
print(X.head())

print("\nTarget:")
print(y.head())


# ============================================================
# 2. TRAIN-TEST SPLIT
# ============================================================

print("\n" + "=" * 70)
print("2. TRAIN-TEST SPLIT")
print("=" * 70)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining data:", X_train.shape)
print("Testing data :", X_test.shape)


# ============================================================
# 3. EVALUATION FUNCTION
# ============================================================

def evaluate_model(model, model_name):

    # Train
    model.fit(X_train, y_train)

    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Metrics
    train_rmse = np.sqrt(
        mean_squared_error(y_train, y_train_pred)
    )

    test_rmse = np.sqrt(
        mean_squared_error(y_test, y_test_pred)
    )

    train_mae = mean_absolute_error(
        y_train,
        y_train_pred
    )

    test_mae = mean_absolute_error(
        y_test,
        y_test_pred
    )

    train_r2 = r2_score(
        y_train,
        y_train_pred
    )

    test_r2 = r2_score(
        y_test,
        y_test_pred
    )

    print(f"\n{model_name}")
    print("-" * 50)

    print(f"Train RMSE : {train_rmse:.4f}")
    print(f"Test RMSE  : {test_rmse:.4f}")

    print(f"Train MAE  : {train_mae:.4f}")
    print(f"Test MAE   : {test_mae:.4f}")

    print(f"Train R²   : {train_r2:.4f}")
    print(f"Test R²    : {test_r2:.4f}")

    return {
        "Model": model_name,
        "Train RMSE": train_rmse,
        "Test RMSE": test_rmse,
        "Train MAE": train_mae,
        "Test MAE": test_mae,
        "Train R2": train_r2,
        "Test R2": test_r2
    }


# ============================================================
# 4. LINEAR REGRESSION
# ============================================================

print("\n" + "=" * 70)
print("3. LINEAR REGRESSION")
print("=" * 70)

results = []

linear_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LinearRegression())
])

linear_result = evaluate_model(
    linear_model,
    "Linear Regression"
)

results.append(linear_result)


# ============================================================
# 5. POLYNOMIAL REGRESSION
# ============================================================

print("\n" + "=" * 70)
print("4. POLYNOMIAL REGRESSION")
print("=" * 70)

polynomial_results = []

degrees = [1, 2, 3, 4, 5]

for degree in degrees:

    model = Pipeline([
        (
            "polynomial",
            PolynomialFeatures(
                degree=degree,
                include_bias=False
            )
        ),

        (
            "scaler",
            StandardScaler()
        ),

        (
            "linear",
            LinearRegression()
        )
    ])

    result = evaluate_model(
        model,
        f"Polynomial Regression Degree {degree}"
    )

    polynomial_results.append(result)


# ============================================================
# 6. POLYNOMIAL DEGREE COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("5. POLYNOMIAL DEGREE COMPARISON")
print("=" * 70)

poly_df = pd.DataFrame(polynomial_results)

print("\n")
print(
    poly_df[
        [
            "Model",
            "Train RMSE",
            "Test RMSE",
            "Train R2",
            "Test R2"
        ]
    ].to_string(index=False)
)


# ============================================================
# 7. BIAS-VARIANCE VISUALIZATION
# ============================================================

degrees_for_plot = [
    1,
    2,
    3,
    4,
    5
]

train_rmse = poly_df["Train RMSE"].values
test_rmse = poly_df["Test RMSE"].values

plt.figure(figsize=(10, 6))

plt.plot(
    degrees_for_plot,
    train_rmse,
    marker="o",
    label="Training RMSE"
)

plt.plot(
    degrees_for_plot,
    test_rmse,
    marker="o",
    label="Testing RMSE"
)

plt.xlabel("Polynomial Degree")
plt.ylabel("RMSE")

plt.title(
    "Bias-Variance Tradeoff: Model Complexity vs Error"
)

plt.xticks(degrees_for_plot)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()


# ============================================================
# 8. RIDGE REGRESSION
# ============================================================

print("\n" + "=" * 70)
print("6. RIDGE REGRESSION")
print("=" * 70)

ridge_results = []

ridge_alphas = [
    0.001,
    0.01,
    0.1,
    1,
    10,
    100
]

for alpha in ridge_alphas:

    ridge_model = Pipeline([

        (
            "polynomial",
            PolynomialFeatures(
                degree=3,
                include_bias=False
            )
        ),

        (
            "scaler",
            StandardScaler()
        ),

        (
            "ridge",
            Ridge(alpha=alpha)
        )
    ])

    result = evaluate_model(
        ridge_model,
        f"Ridge Alpha {alpha}"
    )

    ridge_results.append(result)


# ============================================================
# 9. RIDGE RESULTS TABLE
# ============================================================

ridge_df = pd.DataFrame(ridge_results)

print("\n" + "=" * 70)
print("RIDGE RESULTS")
print("=" * 70)

print(
    ridge_df[
        [
            "Model",
            "Train RMSE",
            "Test RMSE",
            "Train R2",
            "Test R2"
        ]
    ].to_string(index=False)
)


# ============================================================
# 10. RIDGE ALPHA VS TEST ERROR
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    ridge_alphas,
    ridge_df["Test RMSE"],
    marker="o"
)

plt.xscale("log")

plt.xlabel("Ridge Alpha")

plt.ylabel("Test RMSE")

plt.title(
    "Ridge Regularization: Alpha vs Test RMSE"
)

plt.grid(True)

plt.tight_layout()

plt.show()


# ============================================================
# 11. LASSO REGRESSION
# ============================================================

print("\n" + "=" * 70)
print("7. LASSO REGRESSION")
print("=" * 70)

lasso_results = []

lasso_alphas = [
    0.0001,
    0.001,
    0.01,
    0.1,
    1
]

for alpha in lasso_alphas:

    lasso_model = Pipeline([

        (
            "polynomial",
            PolynomialFeatures(
                degree=2,
                include_bias=False
            )
        ),

        (
            "scaler",
            StandardScaler()
        ),

        (
            "lasso",
            Lasso(
                alpha=alpha,
                max_iter=10000
            )
        )
    ])

    result = evaluate_model(
        lasso_model,
        f"Lasso Alpha {alpha}"
    )

    lasso_results.append(result)


# ============================================================
# 12. LASSO RESULTS TABLE
# ============================================================

lasso_df = pd.DataFrame(lasso_results)

print("\n" + "=" * 70)
print("LASSO RESULTS")
print("=" * 70)

print(
    lasso_df[
        [
            "Model",
            "Train RMSE",
            "Test RMSE",
            "Train R2",
            "Test R2"
        ]
    ].to_string(index=False)
)


# ============================================================
# 13. ELASTIC NET
# ============================================================

print("\n" + "=" * 70)
print("8. ELASTIC NET")
print("=" * 70)

elastic_results = []

elastic_alphas = [
    0.001,
    0.01,
    0.1,
    1
]

l1_ratios = [
    0.2,
    0.5,
    0.8
]

for alpha in elastic_alphas:

    for l1_ratio in l1_ratios:

        elastic_model = Pipeline([

            (
                "scaler",
                StandardScaler()
            ),

            (
                "elastic",
                ElasticNet(
                    alpha=alpha,
                    l1_ratio=l1_ratio,
                    max_iter=10000
                )
            )
        ])

        result = evaluate_model(
            elastic_model,
            f"ElasticNet alpha={alpha}, "
            f"l1_ratio={l1_ratio}"
        )

        elastic_results.append(result)


# ============================================================
# 14. ELASTIC NET RESULTS
# ============================================================

elastic_df = pd.DataFrame(elastic_results)

print("\n" + "=" * 70)
print("ELASTIC NET RESULTS")
print("=" * 70)

print(
    elastic_df[
        [
            "Model",
            "Train RMSE",
            "Test RMSE",
            "Train R2",
            "Test R2"
        ]
    ].to_string(index=False)
)


# ============================================================
# 15. CROSS VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("9. CROSS VALIDATION")
print("=" * 70)

cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


def cross_validate_model(model, model_name):

    scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="neg_root_mean_squared_error"
    )

    rmse_scores = -scores

    print(f"\n{model_name}")

    print("Fold RMSE:")
    print(rmse_scores)

    print(
        f"Mean CV RMSE: "
        f"{rmse_scores.mean():.4f}"
    )

    print(
        f"Std CV RMSE : "
        f"{rmse_scores.std():.4f}"
    )

    return {
        "Model": model_name,
        "Mean CV RMSE": rmse_scores.mean(),
        "Std CV RMSE": rmse_scores.std()
    }


cv_results = []


# Linear
cv_linear = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LinearRegression())
])

cv_results.append(
    cross_validate_model(
        cv_linear,
        "Linear Regression"
    )
)


# Ridge
cv_ridge = Pipeline([
    (
        "polynomial",
        PolynomialFeatures(
            degree=3,
            include_bias=False
        )
    ),

    ("scaler", StandardScaler()),

    ("ridge", Ridge(alpha=1))
])

cv_results.append(
    cross_validate_model(
        cv_ridge,
        "Ridge Regression"
    )
)


# Lasso
cv_lasso = Pipeline([
    (
        "polynomial",
        PolynomialFeatures(
            degree=2,
            include_bias=False
        )
    ),

    ("scaler", StandardScaler()),

    (
        "lasso",
        Lasso(
            alpha=0.01,
            max_iter=10000
        )
    )
])

cv_results.append(
    cross_validate_model(
        cv_lasso,
        "Lasso Regression"
    )
)


# Elastic Net
cv_elastic = Pipeline([
    ("scaler", StandardScaler()),

    (
        "elastic",
        ElasticNet(
            alpha=0.01,
            l1_ratio=0.5,
            max_iter=10000
        )
    )
])

cv_results.append(
    cross_validate_model(
        cv_elastic,
        "Elastic Net"
    )
)


cv_df = pd.DataFrame(cv_results)


# ============================================================
# 16. FINAL CV COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("FINAL CROSS-VALIDATION COMPARISON")
print("=" * 70)

print(
    cv_df.to_string(index=False)
)


# ============================================================
# 17. FINAL MODEL
# ============================================================

print("\n" + "=" * 70)
print("10. FINAL MODEL")
print("=" * 70)

final_model = Pipeline([

    (
        "polynomial",
        PolynomialFeatures(
            degree=3,
            include_bias=False
        )
    ),

    (
        "scaler",
        StandardScaler()
    ),

    (
        "ridge",
        Ridge(alpha=1)
    )
])


final_model.fit(
    X_train,
    y_train
)


# Predictions

y_pred = final_model.predict(X_test)


# Metrics

final_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        y_pred
    )
)

final_mae = mean_absolute_error(
    y_test,
    y_pred
)

final_r2 = r2_score(
    y_test,
    y_pred
)


print("\nFinal Model:")
print("Polynomial Degree = 3")
print("Ridge Alpha       = 1")

print(f"\nTest RMSE : {final_rmse:.4f}")
print(f"Test MAE  : {final_mae:.4f}")
print(f"Test R²   : {final_r2:.4f}")


# ============================================================
# 18. ACTUAL VS PREDICTED
# ============================================================

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    y_pred,
    alpha=0.5
)

# Perfect prediction line

minimum = min(
    y_test.min(),
    y_pred.min()
)

maximum = max(
    y_test.max(),
    y_pred.max()
)

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--"
)

plt.xlabel("Actual House Value")

plt.ylabel("Predicted House Value")

plt.title(
    "Actual vs Predicted House Values"
)

plt.grid(True)

plt.tight_layout()

plt.show()


# ============================================================
# 19. RESIDUAL PLOT
# ============================================================

residuals = y_test - y_pred

plt.figure(figsize=(10, 6))

plt.scatter(
    y_pred,
    residuals,
    alpha=0.5
)

plt.axhline(
    y=0,
    linestyle="--"
)

plt.xlabel("Predicted Values")

plt.ylabel("Residuals")

plt.title(
    "Residual Plot - Final Ridge Model"
)

plt.grid(True)

plt.tight_layout()

plt.show()


# ============================================================
# 20. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("PROJECT SUMMARY")
print("=" * 70)

print("""
Bias-Variance Tradeoff:

Low model complexity
        ↓
High Bias
        ↓
Underfitting

High model complexity
        ↓
High Variance
        ↓
Overfitting


Regularization:

Ridge
    → L2 regularization
    → Shrinks coefficients

Lasso
    → L1 regularization
    → Can make coefficients exactly zero

Elastic Net
    → L1 + L2 regularization


Main Learning:

Increasing model complexity:
    Bias decreases
    Variance increases

Increasing regularization:
    Model complexity decreases
    Variance generally decreases
    Bias may increase
""")

print("=" * 70)
print("PROJECT COMPLETED")
print("=" * 70)