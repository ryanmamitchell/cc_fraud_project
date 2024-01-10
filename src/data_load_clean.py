# Data Loading and Cleaning

#   csv file is first loaded into Python
#   analyzing data for outliers, missing values, distributions, etc. is conducted in a jupyter notebook
#   cleaning data by imputing missing values/outliers, normalizing values, and formatting for further modeling

# Packages
import pandas as pd
from sklearn import preprocessing


# Data Load
raw_data = pd.read_csv("data/creditcard.csv")


#Data Clean

# no imputation is necessary as the raw data contains no missing values or outliers
clean_data = raw_data
clean_data.to_csv('data/clean_data.csv')


#Normalization
y_data = raw_data['Class'] # separating x and y variables

x_data = raw_data
x_data.drop('Class', axis=1, inplace=True)
x_norm = x_data.values # converting to array for sklearn format

x_norm = preprocessing.normalize(x_norm, norm='l2') # normalizing x values
x_norm_df = pd.DataFrame(x_norm, index=x_data.index, columns=x_data.columns)

norm_df = x_norm_df.join(y_data) # rejoining x and y
norm_df.to_csv('data/norm_data.csv') # saving to data folder for modeling

