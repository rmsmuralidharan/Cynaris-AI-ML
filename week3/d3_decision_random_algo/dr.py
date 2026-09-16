import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

### importing the dataset
df = pd.read_csv('week3/d2_logistic_regression/data/students_pass_fail.csv')

print(df.head(2))
print(df.shape) 


#### independent and dependent features
x = df.drop(columns=['passed'])
y = df['passed']


### since we have an imabalnced class we'll increase the minority class using smote

smote = SMOTE(random_state=42)

x_resample, y_resample = smote.fit_resample(x, y)

### before over sampling
print('\nbefore over sampling:')
print(y.value_counts())

### after over sampling
print('\nafter over sampling:')
print(y_resample.value_counts())


### train test split

x_train, x_test, y_train, y_test = train_test_split(x_resample, y_resample, test_size=0.2, random_state=42, stratify=y_resample)


### model creation

dt = DecisionTreeClassifier(random_state=42)

## model training

dt.fit(x_train, y_train)


### model prediction

y_pred = dt.predict(x_test)


### model evaluation

score = accuracy_score(y_test, y_pred)

cr = classification_report(y_test, y_pred)

cm = confusion_matrix(y_test, y_pred)

print('\nFinal report for decision tree classifier:')

print(
    f"accuracy score: {score}\n"
    f"Report: {cr}\n"
    f"confusion matrix: {cm}"
)







