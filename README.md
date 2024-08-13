# Stock Selection 

## Overview

This Jupyter notebook contains two sections. Stock selection and ..... 


Requirements
* Python 3.x
  
```python
# libraries to retrieve stock data
import yfinance as yf

# Numpy and Pandas for data manipulation
import pandas as pd
import numpy as np

# Plotting and Visualization
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# sklearn for PCA,kmeans and model metrics
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples

# scipy for Euclidean distance
from scipy.spatial.distance import cdist

```

## First Jupyter notebook - Stock Selection 

The first Jupyter notebook demonstrates a clustering of the S&P 500 stocks data, based on a stock price and volatility. The process  involves the following steps:

## Data Collection:

Fetching Historical Stock Data: The notebook retrieves historical stock price data for S&P 500 companies from Yahoo Finance using the yfinance library. The data consists of 472 companies (excluded 34 companies with less than ten years of trading history) and time range is from July 2008 to July 2024, which allowing this analysis to cover significant financial events such as the Global Financial Crisis, the European Debt Crisis, and the COVID-19 pandemic.

```python
# URL to get S&P tickers from
tickers = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
tickers = tickers['Symbol'].tolist()

# Replace . with - to meet Yahoo Finance ticker's format
tickers = [ticker.replace('.', '-') for ticker in tickers]

# Identidy and filter out stocks with less than ten years of trading history
excluded_tickers = {'SW', 'ENPH','ABNB','CARR','CEG','CRWD',
                    'CTVA','DAY','DOW','ETSY','FOX','FOXA',
                    'FTV','GDDY','GEHC','GEV','HLT','HPE',
                    'HWM','INVH','IR','KHC','KVUE','LW','MRNA',
                    'OTIS','ORVO','SOLV','UBER','VICI','VLTO','VST'}

# Get the list of the ticker names after filter out stocks with less than ten years of trading history
valid_tickers = []
for ticker in tickers:
    if ticker not in excluded_tickers:
        valid_tickers.append(ticker)

```


## Data Preprocessing:

Handling Missing Data: Missing data points are filled using forward and backward filling methods to ensure continuity in the time series data.

Calculation of Financial Metrics: The notebook calculates key financial metrics such as daily returns, quarterly returns, and quarterly volatility. These metrics are essential for understanding the performance and risk profile of each stock.

## Clustering Analysis:

Unsupervised Learning: The notebook applies clustering algorithms (e.g., K-Means) to group stocks based on their calculated financial metrics (e.g., returns and volatility). This helps in identifying patterns and categorizing stocks with similar performance profiles.

Dimensionality Reduction: Techniques such as Principal Component Analysis (PCA) are used to reduce the dimensionality of the data, making the clustering process more efficient and interpretable.

Evaluation of Clusters: The quality of the clusters is evaluated using metrics like the silhouette score, which measures how well each stock fits within its cluster.

## Visualization:
