import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder, StandardScaler, MinMaxScaler, RobustScaler
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_regression

"""
just for the task im using label encoder, because in my dataset there are 5 departments while we do lable encoder
the model will learn it as 5>4>3>2>1, so just for the task im demonstrating the label encoder
but for the final model training i'll use one hot encoder for my department column
"""

## load the dataset

df = pd.read_csv('week2/data/merged.csv')

print(df.head(5))

### label encoding

label_encoder = LabelEncoder()

df['Department_label'] = label_encoder.fit_transform(
    df['Department']
)

print('\nLabel encoding:')
print(df[['Department', 'Department_label']].head(10))

print('\nlabel mapping:')
for category, encoded_value in zip(
    label_encoder.classes_,
    label_encoder.transform(label_encoder.classes_)
):
    print(f'{category} ---> {encoded_value}')




## ordinal encoding

ordinal_encoder = OrdinalEncoder()

df['Department_Ordinal'] = ordinal_encoder.fit_transform(
    df[['Department']]
)

print("\nOrdinal Encoding:")
print(df[["Department", "Department_Ordinal"]].head(10))


## one hot encoding

encoder = OneHotEncoder(
    sparse_output=False,
    handle_unknown='ignore'
)

department_encoded = encoder.fit_transform(
    df[['Department']]
)

onehot_df = pd.DataFrame(
    department_encoded,
    columns=encoder.get_feature_names_out(['Department'])
)

print('\none hot encoding:')
print(onehot_df.head(10))

print("\nData Types:")
print(df.dtypes)

print("\nSample Numeric Columns:")
print(df[[
    "Age",
    "Experience_Years",
    "Salary",
    "Performance_Score"
]].head())


### feature scaling

numeric_features = [
    'Age',
    'Experience_Years',
    'Salary',
    'Performance_Score'
]

## standard scaler
standard_scaler = StandardScaler()

standard_scaled = standard_scaler.fit_transform(
    df[numeric_features]
)

standard_df = pd.DataFrame(
    standard_scaled,
    columns=numeric_features
)

print("\nStandardScaler:")
print(standard_df.head())


## minmax scaler

minmax_scaler = MinMaxScaler()

minmax_scaled = minmax_scaler.fit_transform(
    df[numeric_features]
)

minmax_df = pd.DataFrame(
    minmax_scaled,
    columns=numeric_features
)


print("\nMinMaxScaler:")
print(minmax_df.head())


### robust scaler
robust_scaler = RobustScaler()

robust_scaled = robust_scaler.fit_transform(
    df[numeric_features]
)

robust_df = pd.DataFrame(
    robust_scaled,
    columns=numeric_features
)

print("\nRobustScaler:")
print(robust_df.head())


## plot distribution before and after scaling

for feature in numeric_features:
    plt.figure(figsize=(10,5))

    plt.subplot(1,2,1)
    plt.hist(df[feature], bins=20)
    plt.title(f"before scaling - {feature}")
    plt.xlabel(feature)
    plt.ylabel('Frequency')

    plt.subplot(1,2,2)
    plt.hist(standard_df[feature], bins=20)
    plt.title(f"After standard scaler - {feature}")
    plt.xlabel(feature)
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()



## select k best - top 5 features

x = df[
    [
        'Age',
        'Experience_Years',
        'Salary'
    ]
]

y = df['Performance_Score']

selector = SelectKBest(
    score_func=f_regression,
    k=3
)

x_selected = selector.fit_transform(x,y)

selected_features = x.columns[selector.get_support()]


print("\nSelected Features:")
print(selected_features.tolist())

print("\nFeature Scores:")
for feature, score in zip(x.columns, selector.scores_):
    print(f"{feature} ----> {score}")



# Why the selected features matter

print("\nWhy Selected Features Matter:")

print(
    "Age: Helps capture differences in employee performance "
    "associated with career stage and professional maturity."
)

print(
    "Experience_Years: Strongly related to performance because "
    "greater experience can contribute to skills and job knowledge."
)

print(
    "Salary: Can reflect employee role, responsibility, experience, "
    "and overall position within the organization."
)



# Encoding Trade-offs

print("\nEncoding Trade-offs:")
print("LabelEncoder: Simple integer representation, but may imply an artificial order.")
print("OneHotEncoder: Suitable for nominal categories, but increases the number of features.")
print("OrdinalEncoder: Useful when categories have a meaningful order; not ideal for nominal departments.")
    


