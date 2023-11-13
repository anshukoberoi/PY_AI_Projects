import pandas as pd

# CREATE

# LOAD FROM EXCEL

df=pd.read_csv('churn-bigml-20.csv')

# LOAD FROM DICTIONARY

tempdict={'col1':[1,2,3],'col2':[4,5,6],'col3':[7,8,9]}
dictfd=pd.DataFrame.from_dict(tempdict)

# READ

print(df.head())
print(df.head(10))
print(df.tail())
print(df.tail(5))

# SHOW COLUMNS AND DATATYPES

print(df.columns)
print(df.dtypes)

# SUMMARY

print(df.describe())
print(df.describe(include='object'))
print(df.State.unique())
print(df.Churn.unique())

# FILTERING COLS

print(df.State)
print(df['International plan'])
print(df[['State','International plan']])

# FILTERING ROWS

print(df[df['International plan']=='No'])
print(df[(df['International plan']=='No') &(df['Churn']==False)])

# INDEXING WITH ILOC

print(df.iloc[14])
print(df.iloc[14,0])
print(df.iloc[22:33])

# INDEXING WITH LOC

state=df.copy()
state.set_index('State',inplace=True)
print(state.head())
print(state.loc['OH'])

# UPDATE

# DROP ROWS:

x=df.isnull().sum()
print(x)
df.dropna(inplace=True)
print(x)

# DROP ROWS:

print(df.drop('Area code', axis=1))

# ADD CALCULATED COL

df['New Column'] = df['Total night minutes']+df['Total intl minutes']
print(df.head())

# UPDATE ENTIRE COL

df['New Column'] = 100
print(df.head())

# UPDATE SINGLE VAL

df.iloc[0,-1]=10
print(df.head())

# CONDITION BASED UPDATE USING APPLY
df['Churn Binary'] = df['Churn'].apply(lambda x: 1 if x==True else 0)
print(df.head())

# OUTPUT

df.to_csv('output.csv')
print(df.to_json())
print(df.to_html())

# DELETE

del df
# print(df.head())