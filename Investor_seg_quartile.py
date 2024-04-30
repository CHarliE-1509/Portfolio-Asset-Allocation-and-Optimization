import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from pandas_datareader import data as pdr
import datetime as dt
import yfinance as yf

def inv_seg_qrt(Start_date):

    # Extract Market Data
    # Duration definition

    start= dt.datetime(Start_date)
    end=dt.datetime.now()

    # Data read from CSV
    data1=pd.read_csv("Ticker.csv")
    ticker_data = data1['Company'].tolist()
    ticker_data.append('^NSEI')

    # Scrapping data from yfinance
    yf.pdr_override()
    tickers=ticker_data

    data_df=pd.DataFrame()

    for ticker in tickers:
        df=pdr.get_data_yahoo(ticker,start,end)
        data_df[ticker]=df['Adj Close']


    # Computing average return for each stock
    daily_return=data_df.pct_change(1)
    daily_return=daily_return.dropna()
    #Converting to annual return
    annual_return=daily_return.mean()*252

    # Computing Risk for each stock
    annual_risk=daily_return.std()* math.sqrt(252)

    # Create a Dataframe for Risk and return of each stock
    stock_data=pd.DataFrame()
    stock_data['Expected_Annual_Return']=annual_return
    stock_data['Expected_Annual_risk']=annual_risk
    stock_data['Ticker']=stock_data.index
    stock_data['Return:Risk']=stock_data['Expected_Annual_Return']/stock_data['Expected_Annual_risk']
    # Sort Dataframe
    stock_data.sort_values(by='Return:Risk', axis=0,ascending=False, inplace=False)

    '''We use Efficient Frontier, where for the given rate of risk, we choose only the 
    stock with highest return, therefore removing all other stocks. Market aassumes investors to be
    mean optimisers and will select only max returns yielding stock'''

    remove_stock=[]
    stock_data2=stock_data.copy()

    for ticker in stock_data2['Ticker'].values:
        eff_stock=stock_data.loc[ (stock_data['Expected_Annual_Return']>stock_data['Expected_Annual_Return'][ticker])
                                &(stock_data['Expected_Annual_risk']<stock_data['Expected_Annual_risk'][ticker]) ].empty
        if eff_stock==False:
            remove_stock.append(ticker)

    # Create a duplicate Dataframe icluding stocks to be used in the portfolio
    stock_data2.drop(remove_stock,inplace=True)
    stock_data2.sort_values(by='Return:Risk',axis=0,inplace=True,ascending=False)

    # Building Portfolio
    return1=np.log(data_df/data_df.shift(1)) # Gives log(priice[t]/price[t-1])
    return1.dropna(inplace=True)
    return_port=return1.copy()
    return_port.drop(remove_stock,inplace=True,axis=1)
    return_port.drop('^NSEI',inplace=True,axis=1)

    # Ascertain optimum weight to get max sharpe
    no_portfolios=1000
    weights=np.zeros((no_portfolios,5))
    std=np.zeros(no_portfolios)
    eR=np.zeros(no_portfolios)
    sharpe=np.zeros(no_portfolios)

    return_mean=return_port.mean()
    sigma=return_port.cov()

    for k in range(no_portfolios):
        # Generate random weight vectors
        weight=np.array(np.random.random(5))
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
    weights[max_sharpe,:]
    Stock_weight=weights[max_sharpe,:]

    # Updating weights on equities of the portfolio
    Portfolio_data=stock_data2
    Portfolio_data['Weight']=Stock_weight
    Portfolio_data['Weighted_Return']=Portfolio_data['Weight']*Portfolio_data['Expected_Annual_Return']
    Portfolio_data

    # Annual return of the portfolio
    an_r=Portfolio_data['Weighted_Return'].sum()*100

    return Portfolio_data,an_r