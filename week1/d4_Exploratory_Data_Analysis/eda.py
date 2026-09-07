import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

## load the dataset - Im gonna use day - 3 cleaned dataset for day - 4 eda 
df = pd.read_csv('week1/d3_Data_Loading_Cleaning/data/cleaned_employee_performance.csv')

## confirming that the dataset path actually works
print(df.head(2))

## descriptive statistics - summarizing and describing the important characteristics of my dataset
print('\nDescriptive statistics:')
print(df.describe())

## Dataset Information - to understand the basic structure of my dataset like no. of rows and columns, data types and no. of non-null values

print('\nDataset information:')
print(df.info())

## Missing values
print('\nMissing values:')
print(df.isnull().sum())



## distribution of numeric columns
numeric_columns = [
    'Age',
    'Experience_Years',
    'Salary',
    'Performance_Score'
]

for column in numeric_columns:
    ## creates a new/empty figure
    plt.figure()

    ## creates a histogram(shows us how frequently different values occur)  
    sns.histplot(df[column], kde=True) ## kde - draws a smooth curve of how data is distributed
    plt.title(f'distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.show() ## to display the histo plot


## correlation heatmap - it tell us how strongly two features are related to each other
plt.figure()
sns.heatmap(   ## heatmap represents the correlation values visually (making strong and weak relationships easier to spot)
    df[numeric_columns].corr(),
    annot=True,  # it tells seaborn to enter acutal correlation numbers inside ecah sell
    cmap='coolwarm' ## coloe scheme for the heatmap
)

plt.title('Correlation heatmap')
plt.show()

## top 10 deapartment counts
plt.figure()
df['Department'].value_counts().head(10).plot(kind='bar')
plt.title('top 10 deapartment counts')
plt.xlabel('Department')
plt.ylabel('Employee count')
plt.xticks(rotation = 45)
plt.show()




