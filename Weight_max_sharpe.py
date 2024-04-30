from pandas_datareader import data as pdr
import numpy as np
import datetime as dt
import pandas as pd
import yfinance as yf
import random


#Defining a function to calculate weight
def weight_sharpe(portfolio,start_date):
    start= dt.datetime(start_date)
    end=dt.datetime.now()


    # Define Portfolio
    yf.pdr_override()
    #tickers=['RELIANCE.NS','KOTAKBANK.NS','SBIN.NS','HDFCBANK.NS','INFY.NS','ITC.NS','BHARTIARTL.NS']
    tickers= portfolio #Getting list of portfolio from other strategy page
    data_df=pd.DataFrame()

    for ticker in tickers:
        df=pdr.get_data_yahoo(ticker,start,end)
        data_df[ticker]=df['Adj Close']

    return1=np.log(data_df/data_df.shift(1)) # Gives log(price[t]/price[t-1])
    return1.dropna(inplace=True)

    # Making number of portfolios to analyse what weight gives the best sharpe ratio

    no_portfolios=1000
    weights=np.zeros((no_portfolios,7))
    std=np.zeros(no_portfolios)
    eR=np.zeros(no_portfolios)
    sharpe=np.zeros(no_portfolios)

    return_mean=return1.mean()
    sigma=return1.cov()

    for k in range(no_portfolios):
        # Generate random weight vectors
        weight=np.array(np.random.random(7))
        weight =weight/np.sum(weight)  # Ensuring that sum of weights = 1
        weights[k,:]=weight  # Storing weights into a list
        # Calculate volatility
        std[k]=np.sqrt(np.dot(weight.T,np.dot(sigma,weight)))
        # Calculate expected return
        eR[k]= np.sum(return_mean*weight)
        # Calculate Sharpe ratio
        sharpe[k]=eR[k]/std[k]

    # Optimum weight to achieve maximum sharp ratio
    max_sharpe=sharpe.argmax()
    return weights[max_sharpe,:]
