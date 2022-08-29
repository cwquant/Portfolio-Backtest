# File: allproject.py
# Author: Collin Wendel (cwendel@bu.edu)
# Description: Backtesting a trading strategy

import numpy as np
import pandas as pd
import math
import re
import matplotlib.pyplot as plt



#Helper Functions
def get_security_data(stocktickers, columns=['Date','Adj Close']):
    """ 
    Takes stock ticker list for .csv stock data files as input and returns a pandas dataframe with stock history 
    ASSUMPTIONS: 
    .csv has date index in first column and price in second
    first stockticker is risk-free rate historical data
    """
    data = []
    for i in range(len(stocktickers)):
        stockticker = stocktickers[i]
        check_security_df = pd.read_csv(stockticker + '.csv')
        
        if 'Adj Close' in check_security_df.columns:
            security_df = pd.read_csv(stockticker + '.csv', usecols=columns)
            security_df = security_df.rename(columns={"Adj Close": stockticker + '_Price'})
            security_df.index = security_df['Date']
            security_df = security_df.drop('Date', axis=1)
            
        else:
            security_df = pd.read_csv(stockticker + '.csv')
            security_df.columns = ['Date', stockticker + '_Price']
            security_df.index = security_df['Date']
            security_df = security_df.drop('Date',axis=1)
        if i == 0:
            security_df = security_df.rename(columns={stockticker + '_Price': 'Rf_Price'})
            
        data.append(security_df)
    
    df = pd.concat(data, axis=1, sort=True)
    df = df[:-1]
    df = df.replace('.', np.nan)
    df = df.dropna()
    df['Rf_Price'] = df['Rf_Price'].astype(float)/100
    print(df)
    return df


def get_securitiess_in_portfolio(portfolio):
    """
    Takes a dataframe of a portfolio and returns a list of the stocks and/ or risk-free rate
    ASSUMPTION: uses the get_stock_data function above
    """

    stock = re.compile('(\w+)_Price')
    
    stocks = re.findall(stock, str(portfolio.columns.values))
    
    return stocks


def rand_weights(n):
    ''' Produces n random weights that sum to 1 '''
    k = np.random.uniform(-1,2, n)
    return k /np.sum(k)


def get_returns(portfolio):
    """
    Takes a dataframe of a portfolio with stock and risk-free prices and returns dataframe with the excess returns
    """
    newportfolio = portfolio.copy(deep=True)
    stocks = get_securitiess_in_portfolio(portfolio)
    
    
    for i in range(len(stocks)):
        newportfolio[stocks[i] + '_Returns'] = portfolio[stocks[i] + '_Price'].astype(float).pct_change(1)
        newportfolio = newportfolio.drop([stocks[i] + '_Price'], axis=1)
       
    
    newportfolio = newportfolio.iloc[1:]
    return newportfolio



def get_weights(portfolio, num_portfolios = 100):
  """
  Takes a Dataframe of returns and creates an amount of portfolios equal to num_portfolios (default = 100) with random weighting
  """
  
  stocks = get_securitiess_in_portfolio(portfolio)
  stocks = [x + '_Weight' for x in stocks]
  num_stocks = len(stocks)
  newportfolio = pd.DataFrame(columns=stocks)
  
  

  for row in range(num_portfolios):
    weights = rand_weights(num_stocks)
    newportfolio.loc[row] = weights
  
  
  
  return newportfolio


def get_risky_return_statistics(portfolio, portfoliodata, stocklist):
    """
    takes a dataframe of porfolios with weights and a dataframe with stock returns and calculates the portfolio returns and std for each portfolio
    """
    
    stocks = stocklist
    
    e_r = []
    exreturns = []
    stdevs = []
    for i in range(len(stocks)):
      er = portfoliodata[stocks[i]+ '_Returns'].mean()
      
      e_r.append(er)
    cov = portfoliodata.cov()
    
    returns = [[x] for x in e_r]
    er_df = pd.DataFrame(e_r)
    for i in range(portfolio.shape[0]):
      weights = portfolio.iloc[i].to_numpy()
      
      #Price Matrix
      p = np.asmatrix(returns)
      
      #Weights Matrix
      w = np.asmatrix(weights)
    
      #Covariance Matrix
      C = np.asmatrix(cov.to_numpy())
      
      
      #Filters out portfolios with outlier standard deviations and average returns
      mu = w * p
      sigma = w * (C * w.T)
      sigma = np.sqrt(sigma) 
      if sigma >= 35:
        sigma = np.asmatrix([np.nan])
      
      if mu >= 50:
        mu = np.asmatrix([np.nan])
      if mu <= -10:
        mu = np.asmatrix([np.nan])
      
      exreturns.append(  float(mu.item(0,0)))
      stdevs.append(float(sigma.item(0,0)))
  
    
    portfolio['Daily_Returns'] = exreturns
    portfolio['StDev'] = stdevs
    er_df = er_df.transpose()
    er_df.columns = stocks
    portfolio.dropna(inplace = True)
    
    return portfolio

   

def get_sharpe_ratio(portfolio, prices, stock_list):
    """ 
    takes a portfolio and returns the a new column with the sharp ratio for each weighted portfolio
    """
    rf = prices['Rf_Price'].astype(float).iloc[-1]/365
    for i in range(len(stock_list)):
        if stock_list[i] != 'Rf' and i < 2:
            E_r = portfolio['Daily_Returns']
            stdev = portfolio['StDev']
            sharp_ratio = (E_r - rf) / stdev
            portfolio['Sharpe_Ratio'] = sharp_ratio
    return portfolio


def get_beginning_strategy_weights(portfolio, cutoff = 90):
    #Get Price Data
    cportfolio_df = portfolio
    
    
    # Get Number of securities in portfolio
    c_num_stocks = len(get_securitiess_in_portfolio(cportfolio_df))

    weights = []
    
    #Assume at least 90 days of historical data
    rportfolio_data_df = cportfolio_df[:cutoff].copy(deep = True)
    
    #Remove risk-free asset
    rportfolio_data_df = rportfolio_data_df.drop(['Rf_Price'], axis=1)

    # Get List of Securities in Risky Portfolio
    stock_list = get_securitiess_in_portfolio(rportfolio_data_df)

    #Get Number of securities in Risky Portfolio
    r_num_stocks = len(stock_list)
    
    # Get Risky Portfolio returns
    rportfolio_returns_df = get_returns(rportfolio_data_df)
       
    # Get Portfolios and their respective weights
    risky_portfolios_df = get_weights(rportfolio_data_df, 100)

    # Get Expected Return and StDev for each portfolio
    get_risky_return_statistics(risky_portfolios_df, rportfolio_returns_df, stock_list)

    # Sort by Returns
    risky_portfolios_df.sort_values(by=['Daily_Returns'], inplace = True, ascending=False)

    # Get Sharpe Ratios for Risky Portfolios
    risky_portfolios_df = get_sharpe_ratio(risky_portfolios_df, cportfolio_df, stock_list)

    #Set inf values to zero
    risky_portfolios_df = risky_portfolios_df.replace([np.inf, -np.inf], 0.0)

    # Get the optimal tangent portfolio at max sharpe ratio
    optimal_portfolio_sharpe = risky_portfolios_df['Sharpe_Ratio'].max()
    
    # Sort Portfolios by Sharpe Ratio
    risky_portfolios_df.sort_values(by=['Sharpe_Ratio'], inplace = True, ascending=False)

    #Get the optimal portfolio to use as first day weights in backtest 
    optimal_portfolio_weights = risky_portfolios_df[[stock_list[x] + '_Weight' for x in range(len(stock_list))]].iloc[0]
    
    weights.append(optimal_portfolio_weights)
    weights_df = pd.concat(weights, axis = 1, sort=True)
    
    return weights_df.T

def strategy_backtest(df, cutoff = 90, fees = 0):
    """
    Takes a datafram with historical price data data and bactests trading strategy
    """
    #Get Price Data
    cportfolio_df = df
    
    #Get Stock Returns
    cportfolio_returns_df = get_returns(cportfolio_df)
    
    # Get Number of securities in portfolio
    c_num_stocks = len(get_securitiess_in_portfolio(cportfolio_df))
    
    # Get List of Securities in Risky Portfolio
    stock_list = get_securitiess_in_portfolio(cportfolio_df)

    #Create list for accumulation of dataframes with portfolio weights
    weights = []
    
    #Get beginning portfolio weights based on historical data and training cutoff
    beginning_weights = get_beginning_strategy_weights(cportfolio_df, cutoff)
    
    #Copy dataframe after cutoff day for testing 
    test_df = cportfolio_df[cutoff:].copy(deep = True)
    
    #Add beginning weights
    weights.append(beginning_weights.T)
    
    #Loop over testing day
    for day in range(1,len(test_df)):
        #Set dataframe to only provide the previous day for testing in addition to the historical training data
        rportfolio_data_df = cportfolio_df[:cutoff + day].copy(deep = True)
        
        #Remove risk-free asset
        rportfolio_data_df = rportfolio_data_df.drop(['Rf_Price'], axis=1)
        
        #Print training day (used for debugging)
        print('Test Day = ', day)

        # Get List of Securities in Risky Portfolio
        stock_list = get_securitiess_in_portfolio(rportfolio_data_df)

        #Get Number of securities in Risky Portfolio
        r_num_stocks = len(stock_list)
        
        # Get Risky Portfolio returns
        rportfolio_returns_df = get_returns(rportfolio_data_df)
        
        # Get Portfolios and their respective weights
        risky_portfolios_df = get_weights(rportfolio_data_df, 100)

        # Get Expected Return and StDev for each random portfolio
        get_risky_return_statistics(risky_portfolios_df, rportfolio_returns_df, stock_list)

        # Sort by Returns
        risky_portfolios_df.sort_values(by=['Daily_Returns'], inplace = True, ascending=False)

        # Get Sharpe Ratios for Random Risky Portfolios
        risky_portfolios_df = get_sharpe_ratio(risky_portfolios_df, cportfolio_df, stock_list)

        #Set inf values to zero
        risky_portfolios_df = risky_portfolios_df.replace([np.inf, -np.inf], 0.0)

        # Sort Portfolios by Sharpe Ratio
        risky_portfolios_df.sort_values(by=['Sharpe_Ratio'], inplace = True, ascending=False)

        #Get the optimal portfolio
        optimal_portfolio_weights = risky_portfolios_df[[stock_list[x] + '_Weight' for x in range(len(stock_list))]].iloc[0]
        weights.append(optimal_portfolio_weights)
        opt_w = np.asmatrix(optimal_portfolio_weights.to_numpy())

    #Create dataframe of portfolio weights
    weights_df = pd.concat(weights, axis = 1, sort=False)
    
    #Format dataframe
    weights_df = weights_df.T
    
    #Get Stock Returns for Portfolio Dataframe          #First Day of testing                       #Last Day of testing
    backtest_returns_df = get_returns(cportfolio_df[list(cportfolio_df[:cutoff].index.values)[-1]:list(test_df.index.values)[-1]].iloc[:,1:])
    
    #Get Portfolio Returns
    portfolio_returns = get_risky_return_statistics(weights_df, backtest_returns_df, stock_list)
    
    #Set Dates to index
    portfolio_returns.index = list(test_df.index.values)
    
    #Remove Standard Deviation
    portfolio_returns = portfolio_returns.drop(['StDev'], axis=1)

    #Rename returns to Portfolio Returns for ID
    portfolio_returns = portfolio_returns.rename(columns = {'Daily_Returns': 'Portfolio_Return'})
    
    #Get Portfolio Momentum for Signal Use
    portfolio_returns['Portfolio_Momentum'] = np.where(portfolio_returns['Portfolio_Return'] > 0, 1, -1)
    portfolio_returns['Portfolio_Momentum'] = portfolio_returns['Portfolio_Momentum'].rolling(window=30).mean()
    portfolio_returns['Portfolio_Momentum'] = portfolio_returns['Portfolio_Momentum'].pct_change(1)
    
   
    #Get Market Returns
    market_returns = pd.read_csv('SPY.csv', usecols=['Date','Adj Close'])
    market_returns.index = market_returns['Date']
    market_returns = market_returns.rename(columns={'Adj Close':'Market_Price'})
    market_returns = get_returns(market_returns)
    
    #Get Market Momentum for Signal Use
    portfolio_returns['Market_Return'] = market_returns[list(cportfolio_df[:cutoff].index.values)[-1]:list(test_df.index.values)[-1]].iloc[:,1:].loc[:,'Market_Returns']
    portfolio_returns['Market_Momentum'] = np.where(portfolio_returns['Market_Return'] > 0, 1, -1)
    portfolio_returns['Market_Momentum'] = portfolio_returns['Market_Momentum'].rolling(window=30).mean()
    portfolio_returns['Market_Momentum'] = portfolio_returns['Market_Momentum'].pct_change(1)
    
    #Create empty series for signal and strategy returns
    portfolio_returns['Signal'] = [0 for x in range(len(portfolio_returns))]
    portfolio_returns['Strategy_Return'] = [0 for x in range(len(portfolio_returns))]
    portfolio_value = [0 for x in range(len(portfolio_returns))]
    portfolio_value[0] = 10000
    portfolio_returns['Portfolio_Value'] = portfolio_value
    
    #Reset Index
    portfolio_returns = portfolio_returns.reset_index()
    
    #Create sell portfolio for market signal
    for row in range(1,len(portfolio_returns)):
        if portfolio_returns.loc[:,'Market_Momentum'].iloc[row-1] > portfolio_returns.loc[:,'Portfolio_Momentum'].iloc[row-1]:
            portfolio_returns.loc[row, 'Signal'] = -1
        else:
            portfolio_returns.loc[row, 'Signal'] = 1
    
    #Set Strategy Return to Market Return or Portfolio Return based on signal
    for row in range(len(portfolio_returns)):
        
        if portfolio_returns['Signal'].iloc[row] == 1:
            portfolio_returns.loc[row,'Strategy_Return'] = portfolio_returns.loc[:,'Portfolio_Return'].iloc[row]
        if portfolio_returns['Signal'].iloc[row] == -1:
            portfolio_returns.loc[row,'Strategy_Return'] = portfolio_returns.loc[:,'Market_Return'].iloc[row]
    
    for row in range(1,len(portfolio_returns)):
        if fees == 0:
            portfolio_returns.loc[row, 'Portfolio_Value'] = (1 + portfolio_returns.loc[row,'Strategy_Return']) * portfolio_returns.loc[row-1, 'Portfolio_Value']
        if fees == 1:
            if portfolio_returns.loc[row, 'Signal'] == -1:
                portfolio_returns.loc[row, 'Portfolio_Value'] = (1 + portfolio_returns.loc[row,'Strategy_Return']) * portfolio_returns.loc[row-1, 'Portfolio_Value'] - 20 #$10 for selling portfolio + $10 for buying market
            else:
                portfolio_returns.loc[row, 'Portfolio_Value'] = (1 + portfolio_returns.loc[row,'Strategy_Return']) * portfolio_returns.loc[row-1, 'Portfolio_Value']
            
    #Create Abnormal Returns Series
    portfolio_returns['Abnormal Returns'] = portfolio_returns['Strategy_Return'] - portfolio_returns['Market_Return']

    if fees == 1:
        portfolio_returns['Strategy_Return'] = portfolio_returns['Portfolio_Value'].pct_change(1)
    
    return portfolio_returns

def plot_prices(prices_df):
    """
    Takes a dataframe containing stock prices and plots them
    """
    copy_df = prices_df.copy(deep=True)
    copy_df = copy_df.fillna(0)
    copy_df.plot(title = 'Stock Prices in Portfolio and Market Price (S&P500)')
    plt.show()

def plot_returns(backtest_df):
    """
    Takes a dataframe containing backtest Market Returns, Strategy Returns, and Abnormal Returns and plots the cummulative returns
    """
    copy_df = backtest_df[['Market_Return','Strategy_Return','Abnormal Returns']].copy(deep=True)
    copy_df = copy_df.fillna(0)
    copy_df.plot(title = 'Market, Strategy, & Abnormal Returns')
    plt.show()

def plot_cumulatve_returns(backtest_df):
    """
    Takes a dataframe containing backtest Market Returns, Strategy Returns, and Abnormal Returns and plots the cummulative returns
    """
    copy_df = backtest_df[['Market_Return','Strategy_Return','Abnormal Returns']].copy(deep=True)
    copy_df = copy_df.fillna(0)
    copy_df = copy_df.cumsum(axis=0)
    copy_df.plot(title = 'Cumulative Returns')
    plt.show()

def compute_drawdown(backtest_df):
    """
    which processes a backtested dataframe adn returns the drawdown percentage of strategy and market returns
    """
    copy_df = backtest_df[['Market_Return','Strategy_Return','Abnormal Returns']].copy(deep=True)
    copy_df['Market Prev Max'] = copy_df.loc[:, 'Market_Return'].cummax()
    copy_df['Strategy Prev Max'] = copy_df.loc[:, 'Strategy_Return'].cummax()
    copy_df['Strategy dd_pct'] = (copy_df['Strategy Prev Max'] - copy_df.iloc[:,0])/ copy_df['Strategy Prev Max']
    copy_df['Market dd_pct'] = (copy_df['Market Prev Max'] - copy_df.iloc[:,0])/ copy_df['Market Prev Max']
    print(copy_df)
    return copy_df

def plot_drawdown(dd):
    """
    create and show two charts: 1 - The historical price and previous maximum price. 2 - The drawdown since previous maximum price as a percentage lost.
    """
    dd[['Market dd_pct', 'Strategy dd_pct']].plot(title ='Drawdown Percentage')
    dd[['Market_Return','Market Prev Max', 'Strategy_Return','Strategy Prev Max']].plot(title = 'Maximun Drawdown')
    plt.xlabel('Date')
    plt.show()

# Test code 
if __name__ == '__main__':
        
    # Test Calls
    securities = ['DGS5', 'IWM', 'AMZN','GOOG','NFLX', 'SPY']
    prices = get_security_data(securities)
    plot_prices(prices)

    #Get Portfolio Data
    backtest_df = get_security_data(securities)
    backtest_df = backtest_df.loc['2015-01-01':]

    #To run backtest
    backtest_df = strategy_backtest(backtest_df, 90, fees=1)
    backtest_df_csv = backtest_df.to_csv('Strategy Backtest with Fees.csv')

    #To use example csv for time's sake (ran starting 2015-01-01)
    backtest_df = pd.read_csv('Strategy Backtest.csv')
    plot_returns(backtest_df)
    plot_cumulatve_returns(backtest_df)
    dd_backtest_df = compute_drawdown(backtest_df)
    plot_drawdown(dd_backtest_df)
    backtest_df.describe().to_csv('Summary Statistics.csv')
    