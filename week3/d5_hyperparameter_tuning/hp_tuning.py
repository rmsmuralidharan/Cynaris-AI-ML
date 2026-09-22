import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV



### loading the dataset
df = pd.read_csv('week3/d2_logistic_regression/data/students_pass_fail.csv')

print(df.shape)
print(df.head(2))


### dependent and independent features
x = df.drop(columns=['passed'])
y = df['passed']


## splitting the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

### scaling the data
scaler = StandardScaler()

x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)


### model initialization
knn = KNeighborsClassifier(
    n_neighbors=10
)

### model training
knn.fit(x_train_scaled, y_train)


### model prediction
y_pred = knn.predict(x_test_scaled)

### model evaluation
score = accuracy_score(y_test, y_pred)

cr = classification_report(y_test, y_pred)

cm = confusion_matrix(y_test, y_pred)


# ----------------------
# SVM Classifier
#---------------------

### model initialization

s = SVC(
    kernel='linear',
    C=3
)

### model training
s.fit(x_train_scaled, y_train)

### model predicton
y_pred_svc = s.predict(x_test_scaled)


### model evaluation
score_svc = accuracy_score(y_test, y_pred_svc)

cr_svc = classification_report(y_test, y_pred_svc)

cm_svc = confusion_matrix(y_test, y_pred_svc)



print('\nBefore Hyperparameter tuning report:')

### report
print('\nKNN Classifier final report:')
print(
    f"accuracy score: {score}\n"
    f"classification report:\n {cr}\n"
    f"confusion matrix:\n {cm}"

)


### report
print('\nSupport Vector Classifier final report:')
print(
    f"accuracy score: {score_svc}\n"
    f"classification report:\n {cr_svc}\n"
    f"confusion matrix:\n {cm_svc}"

)


print('\nAfter Hyperparameter tuning report:')

print('\nRandom search CV:')
print('\nFor KNN classifier Model:')

### random searched CV = randomly select the parameters between the range we provide

param_dist = {
    "n_neighbors": range(1, 31),
    "weights": ["uniform", "distance"],
    "metric": ["euclidean", "manhattan", "minkowski"],
    "p": [1, 2]
}

random_search_knn = RandomizedSearchCV(
    estimator=knn,
    param_distributions=param_dist,
    n_iter=10,
    scoring="accuracy",
    cv=5,
    random_state=42,
    n_jobs=-1
)

random_search_knn.fit(x_train_scaled, y_train)

print(f"best params: {random_search_knn.best_params_}")
print(f"best score: {random_search_knn.best_score_}")



param_dist_svc= {
    "C": [0.01, 0.1, 1, 10, 100, 1000],
    "kernel": ["linear", "rbf", "poly", "sigmoid"],
    "gamma": ["scale", "auto"],
    "degree": [2, 3, 4, 5]
}

random_search_svc = RandomizedSearchCV(
    estimator=s,
    param_distributions=param_dist_svc,
    n_iter=10,
    scoring="accuracy",
    random_state=42,
    n_jobs=-1
)

random_search_svc.fit(x_train_scaled, y_train)

print('\nfor SVC model:')
print(f"best params: {random_search_svc.best_params_}")
print(f"best score: {random_search_svc.best_score_}")



#### grid search cv
print('\nGrid Search CV:')

print('\nfor KNN model:')

param_grid = {
    "n_neighbors": [3, 5, 7, 9, 11],
    "weights": ["uniform", "distance"],
    "metric": ["euclidean", "manhattan"],
    "p": [1, 2]
}

grid_search_knn = GridSearchCV(
    estimator=knn,
    param_grid=param_grid,
    scoring='accuracy',
    n_jobs=-1,
    cv=5
)

grid_search_knn.fit(x_train_scaled, y_train)

print(f"best params: {grid_search_knn.best_params_}")
print(f"best score: {grid_search_knn.best_score_}")




print('\nfor SVC model:')

param_grid_svc = {
    "C": [0.1, 1, 10, 100],
    "kernel": ["linear", "rbf", "poly"],
    "gamma": ["scale", "auto"],
    "degree": [2, 3, 4]
}

grid_search_svc = GridSearchCV(
    estimator=s,
    param_grid=param_grid_svc,
    scoring="accuracy",
    n_jobs=-1,
    cv=5
)

grid_search_svc.fit(x_train_scaled, y_train)

print(f"best params: {grid_search_svc.best_params_}")
print(f"best score: {grid_search_svc.best_score_}")






