# Stock Selection 

## Overview

This Jupyter notebook contains two sections: Stock selection and Price Prediction 


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

3. Outlier Reduction: Outliers within the clusters are identified and removed using a threshold-based method. This process calculates pairwise distances within each cluster, determines a threshold based on the maximum and minimum distances, and removes data points that exceed this threshold, ensuring the remaining clusters are more accurate and representative of the underlying data.
4. Stock Selection from Clusters: Within each cluster, returns and volatility are normalized, and a combined score is calculated by summing the normalized return and inverse-normalized volatility. The top five stocks from each cluster are selected based on this combined score, resulting in a curated list of stocks that balances performance and risk across clusters. Industry information is also added into the list.

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

```python
# Read the S&P 500 companies list from Wikipedia
tickers_df = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]

# Extract the 'Symbol' and 'GICS Sector' columns
tickers_df['Symbol'] = tickers_df['Symbol'].str.replace('.', '-')  # Replace '.' with '-' for Yahoo Finance format
tickers_df = tickers_df[['Symbol', 'GICS Sector']]


#create dictionary to include ticker name, cluster number.
ticker_cluster_df = pd.DataFrame({
    'Ticker': reduced_tickers,
    'Cluster': reduced_labels
})

# Consolidate the industry information
consolidated_df = pd.merge(ticker_cluster_df, tickers_df, how='left', left_on='Ticker', right_on='Symbol')

# Select the columns
tickers_clusters_df = consolidated_df[['Ticker', 'Cluster', 'GICS Sector']]
#rename columns
tickers_clusters_df.rename(columns={'GICS Sector': 'Industry'}, inplace=True)
```

```python
# Calculate the average return for each stock
average_return = quarterly_return.mean(axis=1).reset_index()
average_return.columns = ['Ticker', 'Average Return']

# Calculate the average return and volatility for each stock
average_return = quarterly_return.mean(axis=1).reset_index()
average_return.columns = ['Ticker', 'Average Return']
average_volatility = quarterly_volatility.mean(axis=1).reset_index()
average_volatility.columns = ['Ticker', 'Average Volatility']

# Merge the average returns and volatility with the cluster information
consolidated_df = pd.merge(tickers_clusters_df, average_return, on='Ticker')
consolidated_df = pd.merge(consolidated_df, average_volatility, on='Ticker')
```

```python
# Normalized the Data
def normalize(x):
    return (x - x.min()) / (x.max() - x.min())
```

```python
consolidated_df['Normalized Return'] = consolidated_df.groupby('Cluster')['Average Return'].transform(normalize)
consolidated_df['Normalized Volatility'] = consolidated_df.groupby('Cluster')['Average Volatility'].transform(lambda x: (x.max() - x) / (x.max() - x.min()))

# Combine the normalized scores
consolidated_df['Combined Score'] = consolidated_df['Normalized Return'] + consolidated_df['Normalized Volatility']
```

```python
# Select the top five stocks for each cluster based on the combined score
top_stocks_per_cluster_normalized = consolidated_df.groupby('Cluster').apply(lambda x: x.nlargest(5, 'Combined Score')).reset_index(drop=True)
top_stocks_per_cluster_normalized = top_stocks_per_cluster_normalized[['Cluster', 'Ticker', 'Industry', 'Average Return', 'Average Volatility', 'Normalized Return', 'Normalized Volatility', 'Combined Score']]
top_stocks_per_cluster_normalized
```

## Visualization:

1. Visualizing Clusters: Each cluster is represented with ticker symbols for each stock. The use of distinct markers and colors helps in identifying the clusters' distribution across the two principal components.
2. Evaluation of Clusters: The quality of the clusters was evaluated using a silhouette plot, which measures how well each stock fits within its cluster.

```python
# Plot the clusters after outlier reduction
markers = ['P','o', 's', '^', 'H', 'X', '*']
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

plt.figure(figsize=(20, 20))
for cluster_label in np.unique(reduced_labels):
    cluster_indices = np.where(reduced_labels == cluster_label)
    plt.scatter(reduced_data[cluster_indices, 0], reduced_data[cluster_indices, 1], 
                marker=markers[cluster_label], color=colors[cluster_label], s=100, lw=0, alpha=0.7, edgecolor='k', label=f'Cluster {cluster_label}')

# Add the ticker symbol
for i, ticker in enumerate(reduced_tickers):
    plt.annotate(ticker, (reduced_data[i, 0], reduced_data[i, 1]), fontsize=10, alpha=0.8)

plt.title('PCA of S&P 500 Stocks with 7 Clusters')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()
```

```python
# https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html
# Below code from scikit documentation

silhouette_avg = silhouette_score(reduced_data, reduced_labels)
fig, ax1 = plt.subplots(1, 1, figsize=(14, 10))

# The silhouette coefficient can range from -1, 1 but in this example, all the coefficients are positive
ax1.set_xlim([-0.1, 1])
ax1.set_ylim([0, len(reduced_data) + (7 + 1) * 10])

# Compute the silhouette scores for each sample
silhouette_values = silhouette_samples(reduced_data, reduced_labels)

y_lower = 10
for i in range(7):
    ith_cluster_silhouette_values = silhouette_values[reduced_labels == i]
    ith_cluster_silhouette_values.sort()

    size_cluster_i = ith_cluster_silhouette_values.shape[0]
    y_upper = y_lower + size_cluster_i

    color = colors[i]
    ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, facecolor=color, edgecolor=color, alpha=0.7)
    ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

    y_lower = y_upper + 10  # 10 for the 0 samples

ax1.set_title("The silhouette plot for the various clusters.")
ax1.set_xlabel("The silhouette coefficient values")
ax1.set_ylabel("Cluster label")

# The vertical line for average silhouette score of all the values
ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

ax1.set_yticks([])  # Clear the yaxis labels / ticks
ax1.set_xticks(np.arange(-0.1, 1.1, 0.2))

plt.show()
```
