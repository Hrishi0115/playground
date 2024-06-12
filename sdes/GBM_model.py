# Our objective is to model and simulate the future price paths of Apple (AAPL) stock using Stochastic Differential Equations (SDEs),
# specifically the Geometric Brownian Motion (GBM) model.

# 1. Understanding the GBM Model

# The Geometric Brownian Motion (GBM) is defined by the following SDE:
# dS_t = mu * S_t * dt + sigma * S_t * dW_t

# where S_t is the stock price at time t
# mu is the drift coefficient (expected return)
# sigma is the volatility coefficient (standard deviation of returns)
# W_t is a Weiner process (Brownian Motion)

# 2. Fetch Historical Data

import yfinance as yf

# Fetch historical data for Apple (AAPL)

ticker = 'AAPL'

data = yf.download(ticker, start='2022-01-01', end='2023-12-31')

data = data['Adj Close'] # use adjusted close price
# adjusted close price is a stock price that has been modified to include corporate actions such as dividends, stock splits, and new stock offerings
# it provides a more accurate representation of the stock's value over time

# 3. Estimate Parameters (mu and sigma)

import numpy as np

# Log returns, also known as continuously compounded returns, are a way to measure the return of an investment over time.
# They are calculated using the natural logarithm of the ratio of consecutive prices.

# Log returns = log(P_t / P_{t-1})
# where P_t is the price at time t
# P_{t-1} is the price at the previous time period

# Why?

# 1. Additivity
# One of the main advantages of log returns is that they are additive over time. This means that the log return of a multi-period interval is simply the sum of the log returns of the 
# individual periods. E.g. if we have daily log returns, the log return over a year is the sum of the daily log returns.
# This property does not hold for simple returns, which makes log returns easier to work with in many mathematical and statistical contexts.

# 2. Normal Distribution Assumption
# Log returns are often assumed to be normally distributed, which simplifies statistical modelling and analysis. This assumption underlies many financial models, including the Black-Scholes
# option pricing model.
# Simple returns, on the other hand, can exhibit skewness and kurtosis that complicate analysis

# 3. Handling Negative Prices
# Log returns handle negative prices better in the context of financial modelling and avoid issues with compounding negative returns.

# 4. Annualising Returns
# When estimating parameters like drift (mu) and volatility (sigma), log returns allow for straightforward annualisation. The mean of daily log returns can be scaled up by the number
# of trading days in a year (typically 252) to get an annualised drift. Similarly, the standard deviation can be scaled by the square root of the number of trading days to get annualised
# volatility.

# calculate log returns

log_returns = np.log(data / data.shift(1)).dropna()

# `data.shift(1)` shifts the price data down by one period, so each price is compared to the previous period's price
# `data / data.shift(1)` gives the ratio of consecutive prices

# estimate parameters

mu = log_returns.mean() * 252
# This computes the mean of the daily log returns and annualises it by multiplying by 252 (approximate number of trading days in a year). This gives the annualised drift or expected return.
# Expected Return (Estimated mu):
# the expected return (drift) is a measure of the average rate at which an investment is expected to grow over time.
# Interpretation: 
#   this value is annualised, meaning it represents the average growth rate per year
#   the number is in logarthmic terms (log returns), not in direct currency terms like $ 
# To convert log returns to simple returns, you use the exponential function
# Simple Annual Return = e^mu - 1
# E.g. mu = 0.1898
# Simple Annual Return = e^0.1898 - 1 ≈ 0.209 = 20.9% - on average, your investment is expected to grow by approximately 20.9% per year

sigma = log_returns.std() * np.sqrt(252)
# This computes the standard deviation of the daily log returns and annualises it by multiplying it by the square root of 252. This gives the annualised volatility, representing the
# risk or uncertainty of the returns.

# Why multiply by square root of 252?
#   This approach is based on the assumption that returns are independently distributed (i.i.d), which allow us to scale the volatility from a daily basis to an annual basis.
#   The square root rule comes from the properties of the standard deviation in relation to time scaling:
#   For a time series of returns, if the returns are i.i.d., the variance scales with time, and the standard deviation scales with the square root of time.


# Risk (Volatility): The risk or volatility is a measure of how much the returns of an investment are expected to fluctuate or vary over time.
# Interpretation:
#   This value is also annualised and represents the standard deviation of the returns per year.
#   This number indicates the extent to which the returns can deviate from the expected return mu.
#   A higher sigma means more variability and therefore higher risk.
#   A sigma of 0.3692 indicates that the returns can vary by approximately ±36.9% around the expected return annually.
# Volatility (sigma) in log returns is often assumed to be similiar to the volatility in simple returns for small values. However, for completeness and to account for the potential
# difference, we need to adjust it properly.
# sigma_simple = sqrt(e^{sigma^2} - 1)
# E.g. sigma_log = 0.3692
# sigma_simple = sqrt(e^{0.3692^2} - 1) ≈ 0.382 = 38.2%

print(f"Estimated mu: {mu}") # = 0.1899
print(f"Estimated sigma: {sigma}") # = 0.3692

# Practical Example

# Suppose you invest $100 in an asset with the given parameters (estimated mu and sigma)
# Expected Return: On average, your investment is expected to grow by 20.9% per year
#   After one year, your investment might be worth approximately $120.90 ($100 * 1.209)
# Risk: However, because of the volatility, the actual return can vary
#   The returns are expected to fluctuate around this 20.9% average with a standard deviation of 38.2%

# Possible Outcomes:
# One Standard Deviation Range (Approximately 68% Confidence Interval)
# Upper Bound: mu_simple + sigma_simple
# Lower Bound: mu_simple - sigma_simple
# Best-case scenario: If the returns are one standard deviation above the mean:
#   Expected return: 20.9% + 38.2% = 59.1%
#   Your investment could be worth: $100 * (1 + 0.591) = $159.10
# Worst-case scenario: If the returns are one standard deviation below the mean:
#   Expected return: 20.9% - 38.2% = -17.3%
#   Your investment could be worth $100 * (1-0.173) = $82.70

# ... same calculation but for Two Standard Deviations Range (Approximately 95% Confidence Interval)

# 4. Simulate Future Price Paths

# Using the estimated parameters, simulate future price paths.

def simulate_gbm(S0, mu, sigma, T, dt):
    """
    Simulate paths of a Geometric Brownian Motion.

    Parameters:
    S0 : float : Initial stock price
    mu : float : Drift coefficient
    sigma : float : Volatility coefficient
    T : float : Total time (in years)
    dt : float : Time step

    Returns:
    np.array : Simulated paths of stock prices
    """
    N = int(T / dt)
    t = np.linspace(0, T, N)
    W = np.random.standard_normal(size=N)
    W = np.cumsum(W) * np.sqrt(dt)  # Standard Brownian motion
    X = (mu - 0.5 * sigma ** 2) * t + sigma * W
    S = S0 * np.exp(X)
    return S

# Parameters for simulation
S0 = data.iloc[-1]  # Last observed price
T = 1.0  # 1 year
dt = 1/252  # Daily steps
# In the context of simulating price paths using the GBM model, dt represents the time step size, which is the fraction of a year that each step in your simulation represents.
# i.e. dt = 1 / trading_days

# Simulate future price paths
simulated_paths = simulate_gbm(S0, mu, sigma, T, dt)

actual_data = yf.download(ticker, start='2024-01-01', end='2024-06-11')
actual_data = actual_data['Adj Close']

# 5. Visualise the Simulated Paths

import matplotlib.pyplot as plt
import pandas_market_calendars as mcal

# Get the NYSE trading calendar
nyse = mcal.get_calendar('NYSE')

# Get the trading days for 2024
schedule = nyse.schedule(start_date='2024-01-01', end_date='2024-12-31')
trading_days_2024 = schedule.index # trading days for 2024 - 252

# Plot the simulated paths
plt.figure(figsize=(10, 5))
plt.plot(trading_days_2024, simulated_paths, label='Simulated Price Path', color='blue')
plt.plot(actual_data.index, actual_data.values, label='Actual Price', color='red')
plt.xlabel('Time (days)')
plt.ylabel('Price')
plt.title('Simulated Geometric Brownian Motion Paths for AAPL')
plt.show()