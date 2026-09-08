### 1. Label encoder
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder

## loding the mental-health of the employees dataset
df = pd.read_csv('week2/d2_feature_scaling/data/mental_health_dataset.csv')


### 1. Label encoder

label_encoder = LabelEncoder()

labelled = label_encoder.fit_transform(
    df['work_environment']
)

label_df = pd.DataFrame(
    labelled
)

print('\nLabel Encoding:')
for value, label in enumerate(label_encoder.classes_):
    print(f"{label} ----> {value}")


### 2. onehot encoder

categorical_features = [
    'gender',
    'employment_status',
    'work_environment',
    'mental_health_history',
    'seeks_treatment'
]

onehot_encoder = OneHotEncoder(
    sparse_output=False,
    handle_unknown='ignore'
)

print('\nOneHot-Encoding:')

onehot_encoded = onehot_encoder.fit_transform(
    df[categorical_features]
)

onehot_df = pd.DataFrame(
    onehot_encoded,
    columns=onehot_encoder.get_feature_names_out(categorical_features)
)

print(onehot_df.head(5))


### 3. ordinal encoding

ordinal_encoder = OrdinalEncoder(
    categories=[["Low", "Medium", "High"]]
)

ordinal_encoded = ordinal_encoder.fit_transform(
    df[['mental_health_risk']]
)

ordinal_df = pd.DataFrame(
    ordinal_encoded
)

print("\nOrdinal Encoding Mapping:")
for value, category in enumerate(ordinal_encoder.categories_[0]):
    print(f"{category} ----> {value}")
