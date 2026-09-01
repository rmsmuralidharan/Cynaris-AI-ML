import pandas as pd
import os


### load the dataset
df = pd.read_csv('week1/d2_Pandas_Data_Manipulation/data/indian_census.csv')

### basic dataset infomation
print("\ndataset shape:")
print(df.shape)

print('\ndataset data types:')
print(df.dtypes)

print("\nfirst 10 rows")
print(df.head(10))


## filter states with population greater than 50 million
filtered_df = df[df['Population 2011'] > 50000000]

print('\nStates/UTs with population greater than 50 million')
print(filtered_df)
## shape inspection
print(filtered_df.shape)



## group data by category and calculate total population

grouped_df = df.groupby('Category')['Population 2011'].sum()

print('\nTotal population by category:')
print(grouped_df)


### creating a second dataset for merge operation

region_data = {
    "India/State/Union Territory": [
        "Tamil Nadu",
        "Karnataka",
        "Kerala",
        "Maharashtra",
        "Delhi"
    ],
    "Region": [
        "South",
        "South",
        "South",
        "West",
        "North"
    ]
}

region_df = pd.DataFrame(region_data)

## merge census data with region data
merged_df = pd.merge(
    df,
    region_df,
    on='India/State/Union Territory',
    how='inner'
)

print('\nMerged data:')
print(merged_df)


## creating pivot table to analyze population by category and region

pivot_df = pd.pivot_table(
    merged_df,
    values='Population 2011',
    index='Category',
    columns='Region',
    aggfunc='sum'
)

print('\nPivot table')
print(pivot_df)


### export cleaned DataFrame to csv
merged_df.to_csv('cleaned_india_census.csv', index=False)

## export cleaned dataframe to parquet
merged_df.to_parquet('cleaned_india_census.parquet', index=False)


## compare file sizes
csv_size = os.path.getsize('cleaned_india_census.csv')

parquet_size = os.path.getsize('cleaned_india_census.parquet')

print('\nFile size comparision:')
print(f"csv file size: {csv_size} bytes")

print(f"parquet file size: {parquet_size} bytes")

