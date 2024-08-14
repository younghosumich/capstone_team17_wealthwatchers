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

# iterate over the list of ticker names after filter out stocks with less than ten years of trading history and extract the individual tickers
valid_tickers = []
for ticker in tickers:
    if ticker not in excluded_tickers:
        valid_tickers.append(ticker)
```

```python
# Select Yahoo Finance data to retrieve
data = yf.download(valid_tickers, start="2008-07-01", end="2024-07-01")['Adj Close']
```


## Data Preprocessing:

1. Handling Missing Data: Missing data points are filled using forward and backward filling methods to ensure continuity in the time series data.
2. Calculation of Financial Metrics: The notebook calculates key financial metrics such as daily returns, quarterly returns, and quarterly volatility. These metrics are essential for understanding the performance and risk profile of each stock.
3. Data Transformation and Preparation: The financial metrics are cleaned by removing NaN values, transposed the data with tickers as the index, combining the financial metrics into a single DataFrame, and the companies with less than ten years of trading history are excluded from this single DataFrame for further analysis.

```python
# Forward fill and Backward fill the NaN data
data.fillna(method='ffill', inplace=True)  
data.fillna(method='bfill', inplace=True)  
```

```python
#take quarterly average of daily data
daily_return = data.pct_change()
quarterly_return = data.resample('Q').ffill().pct_change()
quarterly_volatility = daily_return.resample('Q').std() * np.sqrt(200)
```

```python
# Forward fill and Backward fill the NaN data for quarterly return and quarterly_volatility
quarterly_return.fillna(method='ffill', inplace=True)
quarterly_return.fillna(method='bfill', inplace=True)
quarterly_volatility.fillna(method='ffill', inplace=True)
quarterly_volatility.fillna(method='bfill', inplace=True)
# Remove the first row from returns and volatility since it is NaN
quarterly_return = quarterly_return.iloc[1:]
quarterly_volatility = quarterly_volatility.iloc[1:]
```

```python
# Transpose the DataFrames
quarterly_return = quarterly_return.transpose()
quarterly_volatility = quarterly_volatility.transpose()
```

```python
# Combine returns and volatility into one DataFrame
features = pd.concat([quarterly_return, quarterly_volatility], axis=1, keys=['Returns', 'Volatility'])
# Remove the excluded tickers from the features DataFrame
features = features.loc[~features.index.isin(excluded_tickers)]
```

## Clustering Analysis:

1. Dimensionality Reduction: Techniques such as Principal Component Analysis (PCA) are used to reduce the dimensionality of the data, making the clustering process more efficient and interpretable.

2. Unsupervised Learning: The notebook applies clustering algorithms (e.g., K-Means) to group stocks based on their calculated financial metrics (e.g., returns and volatility). This helps in identifying patterns and categorizing stocks with similar performance profiles.

3. Evaluation of Clusters: The quality of the clusters is evaluated using metrics like the silhouette score, which measures how well each stock fits within its cluster.

## Visualization:
