import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix



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

### report
print('\nKNN Classifier final report:')
print(
    f"accuracy score: {score}\n"
    f"classification report:\n {cr}\n"
    f"confusion matrix:\n {cm}"

)

