import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn import preprocessing

data=pd.read_csv('housing_price_dataset.csv')

df= pd.DataFrame(data=data)

categorical_features = ['Neighborhood']


# df['Neighborhood'].unique()



# print(df.head())

# print(df['AGE'].unique())


print(df.isnull().sum())



ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False).set_output(transform='pandas')

ohetransformedAGE = ohe.fit_transform(df[['Neighborhood']])
df=pd.concat([df,ohetransformedAGE], axis=1).drop(columns=['Neighborhood'])


price= df['Price'].copy()
df.drop(columns=['Price'], inplace=True)
df['intercept']=1
df['Price']= price



save_path='hpriice_ohe.csv'
df.to_csv(save_path,index=False)




print(df.head())