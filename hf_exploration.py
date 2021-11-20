import pandas as pd
import matplotlib.pyplot as plt

# Read in the data to a dataframe.
df = pd.read_csv('heart.csv')

# Checking the head of the data.
print(df.head())

# Check for null values. We can see we have no missing values.
print(df.loc[df.isnull().any(axis=1)])

# Check some stats of the dataframe.
print(df.describe())

# View distribution of numerical data
fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6)
df.hist(column=['Age'], ax=ax1)
df.hist(column=['RestingBP'], ax=ax2)
df.hist(column=['Cholesterol'], ax=ax3)
df.hist(column=['FastingBS'], ax=ax4)
df.hist(column=['MaxHR'], ax=ax5)
df.hist(column=['Oldpeak'], ax=ax6)
plt.show()

"""
    From the hist plots:
    Age, RestingBP, MaxHR, and Cholesterol appear to be relatively normal dist.
    Cholesterol has one bar at <50 outside the normal dist.
    FastingBS is more of a categorical variable.
    OldPeak is skewed right.
"""

# View correlation
print(df[['Age','RestingBP','Cholesterol','FastingBS','MaxHR','Oldpeak','HeartDisease']].corr())

"""
    Of the above variables, Resting BP appears to have the least correlation and may hinder model.
"""

# View Categorical data
df_cat = df[['Sex','ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope', 'HeartDisease']]

for column in df_cat.columns:
    if column == 'HeartDisease':
        continue
    else:
        print(pd.pivot_table(df_cat, index = column, values='HeartDisease'))

"""
    From the following pivot tables, it may be concluded that RestingECG may not be useful to the 
    model due to most categories being 50% one way or the other.
"""
