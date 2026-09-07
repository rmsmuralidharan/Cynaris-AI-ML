import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

## loading the celaned dataset that i've cleaned on day 3
df = pd.read_csv(
   "week1/d3_Data_Loading_Cleaning/data/cleaned_employee_performance.csv"
)

## display basic info about the dataset
print('dataset shape:', df.shape)
print('\nfirst 5 rows:')
print(df.head(5))

## department wise employee count
plt.figure()
sns.countplot(data=df, x='Department')
plt.title('employee count by department')
plt.xlabel('department')
plt.ylabel('employee count')
plt.xticks(rotation = 45)
plt.tight_layout()
plt.show()

## salary distribution
plt.figure()
sns.histplot(data=df, x='Salary', kde=True)
plt.title('Salary Distribution')
plt.xlabel('Salary')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

## performance wise distribution
plt.figure()
sns.histplot(data=df, x='Performance_Score', kde=True)
plt.title('Performance score Distribution')
plt.xlabel('performance score')
plt.ylabel('Frequencey')
plt.tight_layout()
plt.show()




## correlation heatmap
plt.figure(figsize=(8,6))

correlation = df[
    ['Age', 'Experience_Years', 'Salary', 'Performance_Score']
].corr()

sns.heatmap(
    correlation,
    annot=True,
    cmap='coolwarm',
    fmt='.2f'
)

plt.title('Correlation heatmap')
plt.tight_layout()
plt.show()


## experience vs performance score
plt.figure(figsize=(8,6))

sns.scatterplot(
    data=df,
    x='Experience_Years',
    y='Performance_Score'
)

plt.title('Experience vs Performance_Score')
plt.xlabel('Experience in years')
plt.ylabel('performance score')

plt.tight_layout()
plt.show()