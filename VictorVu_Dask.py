
# # Victor Vu Dask
# 
# For this project, we were given two datasets from Kaggle https://www.kaggle.com/marklvl/bike-sharing-dataset/home containing information about the Bike Sharing service in Washington D.C. "Capital Bikeshare"
# 
# One dataset contains hourly data and the other one has daily data from the years 2011 and 2012.
# 
# The following variables are included in the data:
# 
# * instant: Record index
# * dteday: Date
# * season: Season (1:springer, 2:summer, 3:fall, 4:winter)
# * yr: Year (0: 2011, 1:2012)
# * mnth: Month (1 to 12)
# * hr: Hour (0 to 23, only available in the hourly dataset)
# * holiday: whether day is holiday or not (extracted from Holiday Schedule)
# * weekday: Day of the week
# * workingday: If day is neither weekend nor holiday is 1, otherwise is 0.
# * weathersit: (extracted from Freemeteo)
#     1: Clear, Few clouds, Partly cloudy, Partly cloudy
#     2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
#     3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds
#     4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog
# * temp: Normalized temperature in Celsius. The values are derived via (t-t_min)/(t_max-t_min), t_min=-8, t_max=+39 (only in hourly scale)
# * atemp: Normalized feeling temperature in Celsius. The values are derived via (t-t_min)/(t_max-t_min), t_min=-16, t_max=+50 (only in hourly scale)
# * hum: Normalized humidity. The values are divided to 100 (max)
# * windspeed: Normalized wind speed. The values are divided to 67 (max)
# * casual: count of casual users
# * registered: count of registered users
# * cnt: count of total rental bikes including both casual and registered (Our target variable)
# 
# We are tasked with building a predictive model that can determine how many people will use the service on an hourly basis, therefore we take the first 5 quarters of the data for our training dataset and the last quarter of 2012 will be the holdout against which we perform our validation. Since that data was not used for training, we are sure that the evaluation metric that we get for it (R2 score) is an objective measurement of its predictive power.
# 
# ### Planning
# 
# Initially, we decided to separate the project in 4 steps:
# 
# Data Loading and Exploratory Data Analysis: Load the data and analyze it to obtain an accurate picture of it, its features, its values (and whether they are incomplete or wrong), its data types among others. Also, the creation of different types of plots in order to help us understand the data and make the model creation easier.
# 
# Data Preparation and Feature Engineering: Once we have the data, we would need to prepare it for the modeling stage, standardizing it, changing data types, dropping features, among others. Later, a process of creating features and selecting others based on a number of different criteria like correlation, would also need to be performed.
# 
# Modeling and Tuning: Once we have the data ready, the modeling stage begins, making use of different models (and ensembles) and a strong pipeline with different transformers, we would hopefully produce a model that fits our expectations of performance. Once we have that model, a process of tuning it to the training data would be performed.
# 
# Results and Conclusions: Finally, with our tuned models, we would predict against the test set we decided to separate initially, then plotting those results against their actual values to determine the performance of the model, and finally, outlining our conclusions after this extensive project.
# 
# 
# ### Notes
# 
# For the code to run, you must install the following extensions:
# * Seaborn (aesthetic plots) Version 0.9.0
# * Xgboost (boosting model) Version 0.82
# * Gplearn (genetic features) Version 0.3.0
# 
# The following code performs the task of installing these libraries, if you wish to do so you may uncomment the cell and run it
# 
# Also, a file that is included inside the zip folder called helpers.py is needed to run the code, this file contains the different functions that were created throught the project in a neat folder that declutters the botebook

# In[1]:


# ! pip install seaborn==0.9.0
# ! pip install xgboost==0.82
# ! pip install gplearn==0.3.0


# In[7]:


import warnings
import numpy as np
import pandas as pd
import dask.dataframe as dd
import seaborn as sns
import plotly.tools as tls
import plotly.plotly as py
from sklearn.base import clone
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.decomposition import PCA
from gplearn.genetic import SymbolicTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score as metric_scorer
from sklearn.feature_selection import RFE, SelectFromModel
from sklearn.preprocessing import PolynomialFeatures, KBinsDiscretizer, PowerTransformer
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor

tls.set_credentials_file(username='alejandro321', api_key='yBVtyuhfWpl3rH4TrOGE')
warnings.filterwarnings('ignore')


# In[8]:


SEED = 1
DATA_PATH = 'https://gist.githubusercontent.com/f-loguercio/f5c10c97fe9afe58f77cd102ca81719b/raw/99fb846b22abc8855de305c2159a57a77c9764cf/bikesharing_hourly.csv'
DATA_PATH2 = 'https://gist.githubusercontent.com/f-loguercio/14ac934fabcca41093a51efef335f8f2/raw/58e00b425c711ac1da2fb75f851f4fc9ce814cfa/bikesharing_daily.csv'
PREC_PATH = 'https://gist.githubusercontent.com/akoury/6fb1897e44aec81cced8843b920bad78/raw/b1161d2c8989d013d6812b224f028587a327c86d/precipitation.csv'
TARGET_VARIABLE = 'cnt'
ESTIMATORS = 50


# ### Data Loading
# 
# Here we load the necessary data, print its first rows and describe its contents

# In[9]:


def read_data(input_path):
    return dd.read_csv(input_path, parse_dates=[1])

data = read_data(DATA_PATH)
data_daily = read_data(DATA_PATH2)

data.head()





