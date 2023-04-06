import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.cluster import KMeans
import cvxpy as cp
from scipy.optimize import minimize


# Display all rows and columns for Dataframe
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


# Daily close prices for the given 10 stock tickers
# January 1, 2012- December 21, 2021 (USD)
tickers = ["AAPL", "AMZN", "GOOG", "INTC", "ORCL", "XOM", "CVX", "COP", "HES", "OXY"]
df = yf.download(tickers, start = "2012-01-01", end="2021-12-21", interval = "1d", group_by='ticker')


# AAPL Daily Returns using Close prices
aaplDailyReturns = df['AAPL']['Close'].pct_change().dropna()
                     
# Check if daily returns are normally distributed
# QQ Plot For AAPl Daily Returns
stats.probplot(aaplDailyReturns, dist="norm", plot=plt)
plt.title("QQ Plot of Daily Returns AAPL")
plt.show()

#Shapiro Wilk Test for AAPl Daily Returns
values = stats.shapiro(aaplDailyReturns)
print(values)



# Daily Returns of the 10 stocks
tenStocksDailyReturns = df.xs('Close', level=1, axis=1).pct_change().dropna()

# Correlation of Daily Returns for the 10 stocks
correlation = tenStocksDailyReturns.corr(method='pearson')
print(correlation)



# Quarterly Returns for all 10 stocks Q2 2020 to Q4 2021
dfPandemic = yf.download(tickers, start = "2020-04-01", end="2021-12-31", interval = "1d", group_by='ticker')
tenStocksQuarterlyReturns = dfPandemic.xs('Close', level=1, axis=1).resample('Q').last().pct_change().dropna()

# Tranpose Array because quarterly returns are the features for the stocks
X = tenStocksQuarterlyReturns.T
# K-means clustering, think that the performance was different between two groups of stocks(oil/gas vs tech)
kmeans = KMeans(n_clusters=2)
kmeans.fit(X)

# Scatter Plot of the points and the 2 cluster groups
plt.scatter(X.iloc[:,0], X.iloc[:,1], c=kmeans.labels_)
plt.title("K-Means Cluster Scatterplot")
plt.xlabel("Quarterly Return")
plt.ylabel("Quarterly Return")

# Label the points with Stocks
for i, stock in enumerate(X.index):
    plt.annotate(stock, (X.iloc[i,0], X.iloc[i,1]))

plt.show()



# Optimal Portfolio weights to minimize the variance of portfolio returns
# Global minimum variance portfolio(GMVP)

# Covariance Matrix across the 10 stocks
Q = tenStocksDailyReturns.cov()

# Using CVXPY
# Vector of portfolio weights for 10 stocks
X = cp.Variable(10)

# GMVP
objective = cp.Minimize(cp.quad_form(X,Q))
# All portfolio weights are positive and portfolio weights sum to 1
constraints = [X >= 0, sum(X) == 1]
prob = cp.Problem(objective, constraints)

result = prob.solve()

optimalPortfolioWeights = X.value
print(optimalPortfolioWeights)
print(sum(optimalPortfolioWeights))


# Using scipy
cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
n = len(Q)
bounds = tuple((0,1) for x in range(n))

def GMVP(X,Q):
    return np.dot(X.T, np.dot(Q,X))

# Initialize Weights
x0 = np.ones(n)/n

min_result = minimize(GMVP, x0, args=(Q,), method='SLSQP', bounds=bounds, constraints=cons)

optimalPortfolioWeights_v2 = min_result.x
print(optimalPortfolioWeights_v2)
