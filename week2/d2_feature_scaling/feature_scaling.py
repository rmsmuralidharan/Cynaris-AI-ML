### 1. Label encoder
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder, OrdinalEncoder, StandardScaler, RobustScaler
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_classif

## loding the mental-health of the employees dataset
df = pd.read_csv('week2/d2_feature_scaling/data/mental_health_dataset.csv')

print(df.shape) 
print(df.head(5))


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



### feature scaling

## 1. min-max scaling

numerical_features = [
    'age',
    'physical_activity_days',
    'depression_score',
    'anxiety_score',
    'social_support_score',
    'productivity_score'
]

min_max_scaler = MinMaxScaler()

min_max_scaled = min_max_scaler.fit_transform(
    df[numerical_features]  
)

min_max_df = pd.DataFrame(
    min_max_scaled,
    columns = numerical_features
)

print("\nMin-Max Scaling:")
print(min_max_df.head(5))

### 2. standard scaling

standard_scaler = StandardScaler()

standard_scaled = standard_scaler.fit_transform(
    df[numerical_features]
)

standard_df = pd.DataFrame(
    standard_scaled,
    columns = numerical_features
)

print("\nStandard Scaling:")
print(standard_df.head(5))

### 3. robust scaling
robust_scaler = RobustScaler()

robust_scaled = robust_scaler.fit_transform(
    df[numerical_features]
)   

robust_df = pd.DataFrame(
    robust_scaled,  
    columns = numerical_features
)

print("\nRobust Scaling:")
print(robust_df.head(5))


### 4. plot distribution before and after scaling
"""
for feature in numerical_features:
    plt.figure(figsize=(15, 10))

    plt.subplot(1, 4, 1)
    plt.title(f"Distribution of {feature} before scaling")
    plt.hist(df[feature], bins=20, color='blue', alpha=0.7)
    plt.xlabel(feature)
    plt.ylabel('Frequency')

    plt.subplot(1, 4, 2)
    plt.title(f"Distribution of {feature} after min-max scaling")
    plt.hist(min_max_df[feature], bins=20, color='green', alpha=0.7)
    plt.xlabel(feature)
    plt.ylabel('Frequency')

    plt.subplot(1, 4, 3)
    plt.title(f"Distribution of {feature} after standard scaling")
    plt.hist(standard_df[feature], bins=20, color='red', alpha=0.7)
    plt.xlabel(feature)
    plt.ylabel('Frequency')

    plt.subplot(1, 4, 4)
    plt.title(f"Distribution of {feature} after robust scaling")
    plt.hist(robust_df[feature], bins=20, color='orange', alpha=0.7)
    plt.xlabel(feature) 
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()

"""

### 5. feature selection using SelectKBest

feature_selector = SelectKBest(score_func=f_classif, k=5)

x = df[numerical_features]
y = df['mental_health_risk']

x_selected = feature_selector.fit_transform(x, y)

selected_features = x.columns[feature_selector.get_support()]

print("\nSelected Features using SelectKBest:")
print(selected_features.tolist())

print("\nFeature Scores:")
for feature, score in zip(x.columns, feature_selector.scores_):
    print(f"{feature} ----> {score}")

# Why the selected features matter

print("\nWhy Selected Features Matter:")

print('age: Age can be an important factor in mental health, as different age groups may experience different stressors and challenges that can impact their mental well-being.')

print('physical_activity_days: Regular physical activity has been shown to have a positive impact on mental health, as it can help reduce stress, anxiety, and depression.')

print('depression_score: Depression is a common mental health condition that can significantly impact an individual\'s well-being and quality of life. Monitoring depression scores can help identify individuals who may be at risk and in need of support.')

print('anxiety_score: Anxiety is another prevalent mental health condition that can affect individuals in various ways. Tracking anxiety scores can help identify those who may be struggling and require intervention.')

print('social_support_score: Social support is crucial for mental health, as having a strong support network can help individuals cope with stress and challenges. Assessing social support scores can provide insights into an individual\'s mental well-being.')


