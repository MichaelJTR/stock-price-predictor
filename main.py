import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as dt
from prophet import Prophet

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('../content/all_stocks_5yr.csv')

df.rename(columns={'Name': 'Ticks'})

ticker = 'AMZN'

tick_df = df.loc[df['Ticks'] == ticker.upper()]

tick_df.info()

tick_df.head()

tick_copy = tick_df.copy()
tick_copy.loc[:, 'date'] = pd.to_datetime(tick_copy.loc[:, 'date'], format="%Y/%m/%d")

tick_copy.info()

tick_df.isnull().sum()

# Simple plotting of Ticker Stock Price
# First Subplot
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,5))
ax1.plot(tick_copy["date"], tick_copy["close"])
ax1.set_xlabel("Date", fontsize=12)
ax1.set_ylabel("Stock Price")
ax1.set_title(f"{ticker} Close Price History")

# Second Subplot
ax1.plot(tick_copy["date"], tick_copy["high"], color="green")
ax1.set_xlabel("Date", fontsize=12)
ax1.set_ylabel("Stock Price")
ax1.set_title(f"{ticker} High Price History")

# Third Subplot
ax1.plot(tick_copy["date"], tick_copy["low"], color="red")
ax1.set_xlabel("Date", fontsize=12)
ax1.set_ylabel("Stock Price")
ax1.set_title(f"{ticker} Low Price History")

# Fourth Subplot
ax2.plot(tick_copy["date"], tick_copy["volume"], color="orange")
ax2.set_xlabel("Date", fontsize=12)
ax2.set_ylabel("Stock Price")
ax2.set_title(f"{ticker}'s Volume History")
plt.show()

m = Prophet()

# Drop the columns
ph_df = tick_copy.drop(['open', 'high', 'low','volume', 'Ticks'], axis=1)
ph_df.rename(columns={'close': 'y', 'date': 'ds'}, inplace=True)

ph_df.head()

m.fit(ph_df)

# Create Future dates
future_prices = m.make_future_dataframe(periods=365)

# Predict Prices
forecast = m.predict(future_prices)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

# Dates
starting_date = dt.datetime(2018, 4, 7)
starting_date1 = mdates.date2num(starting_date)
trend_date = dt.datetime(2018, 6, 7)
trend_date1 = mdates.date2num(trend_date)

pointing_arrow = dt.datetime(2018, 2, 18)
pointing_arrow1 = mdates.date2num(pointing_arrow)

# Learn more Prophet tomorrow and plot the forecast for amazon.
fig = m.plot(forecast)
ax1 = fig.add_subplot(111)
ax1.set_title(f"{ticker} Stock Price Forecast", fontsize=16)
ax1.set_xlabel("Date", fontsize=12)
ax1.set_ylabel("Close Price", fontsize=12)

# Forecast initialization arrow
ax1.annotate('Forecast \n Initialization', xy=(pointing_arrow1, 1350), xytext=(starting_date1,1700),
            arrowprops=dict(facecolor='#ff7f50', shrink=0.1),
            )

# Trend emphasis arrow
ax1.annotate('Upward Trend', xy=(trend_date1, 1225), xytext=(trend_date1,950),
            arrowprops=dict(facecolor='#6cff6c', shrink=0.1),
            )

ax1.axhline(y=1260, color='b', linestyle='-')

plt.show()

fig2 = m.plot_components(forecast)
plt.show()

# Monthly Data Predictions
m = Prophet(changepoint_prior_scale=0.01).fit(ph_df)
future = m.make_future_dataframe(periods=12, freq='M')
fcst = m.predict(future)
fig = m.plot(fcst)
plt.title("Monthly Prediction \n 1 year time frame")

plt.show()

fig = m.plot_components(fcst)
plt.show()