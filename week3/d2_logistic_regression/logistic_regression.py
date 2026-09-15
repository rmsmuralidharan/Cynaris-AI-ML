import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

## loading the dataset
df = pd.read_csv('week3/d2_logistic_regression/data/students_pass_fail.csv')

print(df.shape)
print(df.head(5))


### data validation
print('\nmissing values:')
print(df.isnull().sum())


print('\nduplicate values:')
print(df.duplicated().sum())


print('\ndata types:')
print(df.dtypes)

### independent and dependent features
x = df.drop(columns=['passed'], axis=0)

y = df['passed']


### train test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=42)

## feature scaling
scaler = StandardScaler()

x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

### model training
logistic = LogisticRegression()

logistic.fit(x_train_scaled, y_train)


### model prediction

y_pred = logistic.predict(x_test_scaled)

### probability of class 1
y_prob = logistic.predict_proba(x_test_scaled)
print(y_prob[:10])


