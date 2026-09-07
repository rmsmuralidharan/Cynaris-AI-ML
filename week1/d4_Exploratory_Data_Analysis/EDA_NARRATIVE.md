## eda narrative

"""
The dataset contains 120 employee records across 6 columns, including department, age, expereince, salary and performance score. the data hase already been cleaned with no missing or duplicate values remaining.

The age distribution ranges from 22 to 40 years with an average age of about 31.6 years. Employees have between 0 and 15 years of experience with an average of approximately 7.4 years. Salary ranges from 35,482 to 119,696 with and average salary of about 77,283. Performance scores ranges from 55 to 99, with and average score of approximately 75.7.

The distribution plots show that the numerical variables are spread across their
ranges rather than being concentrated at a single value. The correlation heatmap
shows weak relationships between the numerical variables. The strongest observed
relationship is between experience and salary, with a correlation of approximately
-0.18. This weak negative correlation is somewhat unexpected because salary would
often be expected to increase with experience, making this relationship worth
investigating further.

The department count plot shows that ML has the highest number of employees,
followed by MLOps and Data. The dataset does not show obvious extreme values from
the basic distributions, but further analysis could investigate salary differences
between departments and whether performance varies by experience or department.
"""