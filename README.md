# Overview
The objective of the project is to use both unsupervised and supervised machine learning techniques to select and build a stock portfolio that provides promising returns. First we selected 15 stocks using Clustering, PCA and Sillhouette score and then used created an LSTM model that predicts the stock price of the selected stocks. Finally we create a portfolio using mean-variance theory and compare the performance of the porftolio against an equal weighted benchmark and the S&P 500.

# Features
- Dimensionality Reduction: Principal Component Analysis (PCA) is used to reduce the dimensionality of the data, making the clustering process more efficient and interpretable.
- Clustering Analysis: The notebook applies clustering algorithms (K-Means) to group stocks based on their calculated financial metrics (quartely returns and volatility). This helps in identifying patterns and categorizing stocks with similar performance profiles.
- Outlier Reduction: Outliers within the clusters are identified and removed using a threshold-based method. This process calculates pairwise distances within each cluster, determines a threshold based on the maximum and minimum distances, and removes data points that exceed this threshold, ensuring the remaining clusters are more accurate and representative of the underlying data.
- Stock Selection from Clusters: Within each cluster, returns and volatilities are normalized, and a combined score is calculated by summing the normalized return and inverse-normalized volatilities. The top five stocks from each cluster are selected based on their combined score, which resulting in a curated list of stocks that balances performance and risk across clusters. 
- LSTM (Long Short Term Memory) prediction model: This model looks at the previous 60 days closing price to predict the next close price of a given stock.
- Portfolio Optimzation: weights for the selected stocks are calculated based on mean variance theory, but using the predicted price instead of the actual stock price.
 
# Requirements and Installation

Steps to install and set up the project:

```python
# clone the repository
git clone https://github.com/younghosumich/capstone_team17_wealthwatchers.git

# Change to our project directory
cd capstone_team17_wealthwatchers

# install requirements
pip install -r requirements.txt
```

# Executing the Program (Step by Step)
1. Run the Data exploration.ipynb to load and perform exploratory data analysis
2. Run the Unsupervised Learning.ipynb to perform cluserting analysis and resulting in the stock selection.
3. Run the Supervised_Learning.ipynb to perform LSTM model building and optimal portfolio construction.



# Authors
Paul R. Boothroyd III,
Wnexiong Ye,
Youngho Shin,
Yeonjae Choo
