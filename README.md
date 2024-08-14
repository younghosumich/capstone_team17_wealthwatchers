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

```python
# Standardize the features data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(features)
# Apply PCA
pca = PCA(n_components=2)
pca_components = pca.fit_transform(scaled_data)
```

```python
# Apply K-mean
# K-means will produce different clusters for different initializations of the cluster centers
# Run k-means with multiple (random) initializations of clusters centers. Take the clustering with the lowest loss
n_init = 100 
kmeans = KMeans(n_clusters=7, n_init=n_init, random_state=26)
kmeans.fit_predict(pca_components)
kmeans_labels = kmeans.labels_
```

```python
# Compute pairwise distances within a cluster
def compute_pairwise_distances(data):
    return cdist(data, data, 'euclidean')

# Reduce outliers based on the threshold using pairwise distance (max + min)/2, and then using that to calculate the Euclidean distance of all the data and removing any data that exceeds the threshold
def reduce_outliers(data, labels, threshold_factor=1):
    new_data = []
    new_labels = []
    valid_indices = []

    # Loop over each cluster
    for cluster_label in np.unique(labels):
        cluster_indices = np.where(labels == cluster_label)[0]
        cluster_data = data[cluster_indices]
        # No outlier if a cluster has 1 or fewer data points
        if len(cluster_data) <= 2:
            new_data.append(cluster_data)
            new_labels.extend([cluster_label] * len(cluster_data))
            valid_indices.extend(cluster_indices)
        else:
            # Compute the threshold for outlier detection
            pairwise_distances = compute_pairwise_distances(cluster_data)
            max_distance = np.max(pairwise_distances)
            min_distance = np.min(pairwise_distances)
            threshold = threshold_factor * (max_distance + min_distance) / 2
            # Compute the distances from the centroid
            centroid = np.mean(cluster_data, axis=0)
            distances_from_centroid = np.linalg.norm(cluster_data - centroid, axis=1)

            # Keep the non-outlier data
            for i, distance in enumerate(distances_from_centroid):
                if distance <= threshold:
                    new_data.append(cluster_data[i])
                    new_labels.append(cluster_label)
                    valid_indices.append(cluster_indices[i])

    return np.array(new_data), np.array(new_labels), valid_indices
```

```python
# Reduce outliers for cluster data
reduced_data, reduced_labels, valid_indices = reduce_outliers(pca_components, kmeans_labels, threshold_factor=1)

# Get the list of tickers of the reduced data
reduced_tickers = features.index[valid_indices]
```
## Visualization:
