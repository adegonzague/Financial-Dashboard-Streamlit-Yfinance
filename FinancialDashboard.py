
"""
This is my first dashboard. 
Created on Sat Oct 15 16:05:46 2022
@author: adegonzague
"""

#==============================================================================
# Initiating
#==============================================================================

import streamlit as st
import pandas as pd
import yfinance as yf
from PIL import Image
import datetime as dt
from datetime import datetime, timedelta
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np 
import matplotlib.pyplot as plt
import pandas_datareader.data as web

#==============================================================================
# Summary
#==============================================================================

def tab1():
    
    # Dashboard title and data description
    st.title("Summary")
    st.write("Data source: Yahoo Finance. URL: https://finance.yahoo.com/")
    
    ## Part 1
    st.write('**1. Business Summary:**')
    
    @st.cache
    def GetCompanyInfo(ticker):
        return yf.Ticker(ticker).info
    
    # Display every ticker's logo
    for tick in tickers:     
       col1, col2 = st.columns([1, 7])
       info = GetCompanyInfo(tick)
       col1.image(info['logo_url'])
       col2.write("")
       shareholders = yf.Ticker(tick).institutional_holders
       shareholders = shareholders.set_index('Holder')
       shareholders = shareholders[:5]
       
       # Create Expander to display business summary and major shareholders for every ticker
       with col2.expander("Business Summary"):
           st.write(info['longBusinessSummary'])
           st.subheader("Top 5 Shareholders:")
           st.dataframe(shareholders)

    ## Part 2
    st.write('**2. Key Statistics:**')
    
    
    # Create Quote Table with financial information by ticker per line
    quote_table = pd.DataFrame()
    for tick1 in tickers:
        i = tickers.index(tick1)
        tick = yf.Ticker(tick1)
        quote_table.loc[i, "tick"] = tick1
        quote_table.loc[i, "Previous Close"] = tick.info['previousClose']
        quote_table.loc[i, "Open"] = tick.info['open']
        quote_table.loc[i, "Bid"] = tick.info['bid']
        quote_table.loc[i, "Ask"] = tick.info['ask']
        quote_table.loc[i, "Day Low"] = tick.info['dayLow']
        quote_table.loc[i, "Day High"] = tick.info['dayHigh']
        quote_table.loc[i, "52 Week Range"] = tick.info['52WeekChange']
        quote_table.loc[i, "Volume"] = tick.info['volume']
        quote_table.loc[i, "Avg. Volume"] = tick.info['averageVolume']
        quote_table.loc[i, "Market Cap"] = tick.info['marketCap']
        quote_table.loc[i, "Beta (5Y Monthly)"] = tick.info['beta']
        quote_table.loc[i, "PE Ratio (TTM)"] = tick.info['trailingPE']
        quote_table.loc[i, "EPS (TTM)"] = tick.info['trailingEps']
        quote_table.loc[i, "Forward Dividend & Yield"] = tick.info['dividendYield']
        quote_table.loc[i, "1y Target Est"] = tick.info['targetMeanPrice']
    # Set tick as index to get rid of the index column
    quote_table = quote_table.set_index('tick')
    # Display Table
    st.dataframe(quote_table)
    
    ## Part 3
    st.write('**3. Chart:**')
    
    # Create a selectbox assigned to daterange, to be used to GetStockData1
    global daterange
    daterange = st.selectbox("Date Range", ('1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'))
    
    # Function to get the data to be plot later on
    @st.cache
    def GetStockData1(tickers, daterange):
        stock_price = pd.DataFrame()
        for tick in tickers:
            stock_df = yf.Ticker(tick).history(period = daterange, interval = "1d")
            stock_df['Ticker'] = tick
            stock_price = pd.concat([stock_price, stock_df], axis=0)
        return stock_price.reset_index()
    
    # Initiate the plot
    fig = go.Figure()
    
    # Initiate the subplot to manage two y axis with the same x axis
    fig = make_subplots(specs=[[{"secondary_y": True}]], 
                        shared_xaxes=True)
    
    # Loop over tick to create traces per tick and assign a color with i
    for i, tick in enumerate(tickers):
          stock_price = GetStockData1([tick], daterange)
          df = stock_price[stock_price['Ticker'] == tick]
          
          # Create Scatter plot with the secondary y axis
          fig.add_trace(go.Scatter(x=df['Date'], 
                                   y=df['Close'], 
                                   name = tick),
                        secondary_y = True) 
         
          # Create Bar Chart with the primary y axis 
          fig.add_trace(go.Bar(x=df['Date'], 
                               y=df['Volume'], 
                               name = tick),
                        secondary_y = False)
          
          # Assign color per tick for both traces
          fig.update_traces(marker=dict(color=fig.layout['template']['layout']['colorway'][i]),
                            selector=dict(name=tick))
          
    # Update layout to make it easier to read and change yaxis1 to make the barchart small compare to the Scatter plot
    fig.update_layout( 
      title={'text': "Stock Prices",
             'y':0.9,
             'x':0.5,
             'xanchor': 'center',
             'yanchor': 'top'},
      #have the info of both traces in one box when we hover over the graph
      hovermode="x unified",
      xaxis={'title':'Date'},
      yaxis1 = dict(title="Volume", range=[0, 10**9]),
      yaxis2={'title':'Closing Price'})
    fig.layout.yaxis.showgrid=False
    
    # Display Chart
    st.plotly_chart(fig, use_container_width=True)
        

#==============================================================================
# Chart
#==============================================================================

def tab2():
    
    # Dashboard title and data description
    st.title("Chart")
    st.write("Data source: Yahoo Finance. URL: https://finance.yahoo.com/")
    
    # Create two columns to display all the select boxes and radio abd the checkbox shw_SMA
    col1, col2= st.columns(2)
    
    global daterange
    daterange = col1.selectbox("Date Range", ('3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'))
    
    global interval
    interval = col2.selectbox("Interval", ('1d', '5d', '1wk', '1mo' ,'3mo'))
    
    chart_type = col1.radio('Plot type',['Line','Candle'], horizontal=True)
    
    # Get Data based on Daterange AND interval
    @st.cache
    def GetStockData2(tickers, daterange = "3m", interval = "1d"):
        stock_price = pd.DataFrame()
        for tick in tickers:
            stock_df = yf.Ticker(tick).history(period = daterange, interval = interval)
            stock_df['Ticker'] = tick
            #stock_df['SMA'] = stock_df.Close.rolling(50).mean()
            stock_price = pd.concat([stock_price, stock_df], axis=0)
        return stock_price.reset_index()
    
    col2.caption('Check to see the')
    shw_SMA = col2.checkbox("50 Days Moving Average") 
    
    # Create Line chart if "Line" is checked on the dashboard

    if chart_type == "Line":
        
        fig = go.Figure()
        fig = make_subplots(specs=[[{"secondary_y": True}]], shared_xaxes=True)
        
        for i, tick in enumerate(tickers):
              stock_price = GetStockData2([tick], daterange, interval)
              df = stock_price[stock_price['Ticker'] == tick]
              
              fig.add_trace(go.Scatter(x=df['Date'], 
                                       y=df['Close'], 
                                       name = tick), 
                            secondary_y = True) 
              
              fig.add_trace(go.Bar(x=df['Date'], 
                                   y=df['Volume'], 
                                   name = tick), 
                            secondary_y = False)
              
              # Add the trace SMA if shw_SMA is checked on the dashboard
              if shw_SMA:
                        fig.add_trace(go.Scatter(x=df['Date'], 
                                                       y= df['Close'].rolling(50).mean(), 
                                                       mode='lines', 
                                                       name = '50-day SMA',
                                                       line = dict(color= fig.layout['template']['layout']['colorway'][i], dash='dash')), 
                                                 secondary_y = True)
              
            
              fig.update_traces(
              marker=dict(color=fig.layout['template']['layout']['colorway'][i]), 
              selector=dict(name=tick)) 
              

     # Create Candlestick chart if "Candle" is checked on the dashboard
        
    else:
        fig = go.Figure()
        fig = make_subplots(specs=[[{"secondary_y": True}]], 
                            shared_xaxes=True)
        
        for i, tick in enumerate(tickers):
              stock_price = GetStockData2([tick], daterange, interval)
              df = stock_price[stock_price['Ticker'] == tick]
              
              # Create Candlestick figure
              fig.add_trace(go.Candlestick(x=df['Date'], #https://plotly.com/python/candlestick-charts/
                                            open=df['Open'],
                                            high=df['High'],
                                            low=df['Low'],
                                            close=df['Close'], 
                                            name = tick),
                            secondary_y = True)
        
              # Change colors of the candlesticks so that it adapt to every tick when there is multiple ones
              fig.update_traces(increasing_line_color= fig.layout['template']['layout']['colorway'][i], # https://community.plotly.com/t/apply-color-groups-for-multiple-traces/59950/2
                                decreasing_line_color= fig.layout['template']['layout']['colorway'][i+5], 
                                selector=dict(name=tick))
              
              # Add bar chart with the same colors
              fig.add_trace(go.Bar(x=df['Date'], 
                                   y=df['Volume'], 
                                   name = tick,
                                   marker=dict(color=fig.layout['template']['layout']['colorway'][i])),
                            secondary_y = False)
              
              # Create Line chart if "Line" is checked on the dashboard
              if shw_SMA:
                          fig.add_trace(go.Scatter(x=df['Date'], 
                                                         y= df['Close'].rolling(50).mean(), 
                                                         mode='lines', 
                                                         name = '50-day SMA',
                                                         line = dict(color= fig.layout['template']['layout']['colorway'][i], dash='dash')), 
                                                   secondary_y = True)
              
    fig.update_layout(
              autosize = False,
              # Set height to make the candlestick chart easier to read
              height=800,
              title={'text': "Stock Prices",
             'y':0.9,
             'x':0.5,
             'xanchor': 'center',
             'yanchor': 'top'},
      hovermode="x unified",
      xaxis={'title':'Time'},
      yaxis1 = dict(title="Volume", range=[0, 10**9]),
      yaxis2 = {'title':'Stock Price'} )
    fig.layout.yaxis.showgrid=False
    
    # Display Chart
    st.plotly_chart(fig, use_container_width=True)
    
#==============================================================================
# Financials
#==============================================================================

def tab3():
    
    # Dashboard title and data description
    st.title("Financials")
    st.write("Data source: Yahoo Finance. URL: https://finance.yahoo.com/")
    
    # Create two columns for selectbox financial statement and period radio
    col1, col2 = st.columns(2)
    FinancialStatement = col1.selectbox("Select Financial Statement", ["Income Statement", "Balance Sheet", "Cash Flow"])
    Period = col2.radio("Select Period", ["Yearly", "Quaterly"], horizontal=True)
    
    # Create the right table depending on the FinancialStatement selected AND the Time Period
    def FinancialReport(FinancialStatement,Period): 
            if (FinancialStatement == 'Income Statement') & (Period == 'Yearly'):
                Report = tick.financials
            elif (FinancialStatement == 'Income Statement') & (Period == 'Quaterly'):
                Report = tick.quarterly_financials
            elif (FinancialStatement == 'Balance Sheet') & (Period == 'Yearly'):
                Report = tick.balance_sheet
            elif (FinancialStatement == 'Balance Sheet') & (Period == 'Quaterly'):
                 Report = tick.quarterly_balance_sheet
            elif (FinancialStatement == 'Cash Flow') & (Period == 'Yearly'):
                Report = tick.cashflow
            elif (FinancialStatement == 'Cash Flow') & (Period == 'Quaterly'):
                Report = tick.quarterly_cashflow
            # Simplify date columns to make it easier to read
            Report.columns =  Report.columns.date 
            return st.table(Report)
    
    # Display the right table with a title
    for tick in tickers:
        st.subheader(str(tick) + ' : ' + str(Period) + ' ' + str(FinancialStatement))
        tick = yf.Ticker(tick)
        FinancialReport(FinancialStatement,Period)
        
#==============================================================================
# Monte Carlo
#==============================================================================
      
def tab4():
      
    # Dashboard title and data description
    st.title("Monte Carlo")
    st.write("Data source: Yahoo Finance. URL: https://finance.yahoo.com/")
    
    # Create two columns for the two selectbox
    col1, col2 = st.columns(2)
    Times = col1.selectbox("Number of Simulations", [200, 500, 1000])
    Days = col2.selectbox("Time Horizon", [30, 60, 90])

    
    # Code from class, slightly adapted
    class MonteCarlo(object):
        
        def __init__(self, tick, data_source, time_horizon, n_simulation, seed):
            
            # Initiate class variables
            self.tick = tick  # Stock ticker
            self.data_source = data_source  # Source of data, e.g. 'yahoo'
            self.end_date = datetime.now().date() # Today's Date
            self.start_date = self.end_date - timedelta(days = Times) # Difference between today and the time horizon selected
            self.time_horizon = Days  # Days
            self.n_simulation = Times  # Number of simulations
            self.seed = seed  # Random seed
            self.simulation_df = pd.DataFrame()  # Table of results
            
            # Extract stock data
            self.stock_price = web.DataReader(tick, data_source, self.start_date, self.end_date)
            
            # Daily return (of close price)
            self.daily_return = self.stock_price['Close'].pct_change()
            # Volatility (of close price)
            self.daily_volatility = np.std(self.daily_return)
            
        def run_simulation(self):
            
            # Run the simulation
            np.random.seed(self.seed)
            self.simulation_df = pd.DataFrame()  # Reset
            
            for i in range(self.n_simulation):
    
                # The list to store the next stock price
                next_price = []
    
                # Create the next stock price
                last_price = self.stock_price['Close'][-1]
    
                for j in range(self.time_horizon):
                    
                    # Generate the random percentage change around the mean (0) and std (daily_volatility)
                    future_return = np.random.normal(0, self.daily_volatility)
    
                    # Generate the random future price
                    future_price = last_price * (1 + future_return)
    
                    # Save the price and go next
                    next_price.append(future_price)
                    last_price = future_price
    
                # Store the result of the simulation
                next_price_df = pd.Series(next_price).rename('sim' + str(i))
                self.simulation_df = pd.concat([self.simulation_df, next_price_df], axis=1)

        def plot_simulation_price(self):
            
            # Plot the simulation stock price in the future
            fig, ax = plt.subplots()
            fig.set_size_inches(15, 10, forward=True)
    
            plt.plot(self.simulation_df)
            plt.title('Monte Carlo simulation for ' + self.tick + \
                      ' stock price in next ' + str(self.time_horizon) + ' days')
            plt.xlabel('Day')
            plt.ylabel('Price')
    
            plt.axhline(y=self.stock_price['Close'][-1], color='red')
            plt.legend(['Current stock price is: ' + str(np.round(self.stock_price['Close'][-1], 2))])
            ax.get_legend().legendHandles[0].set_color('red')
    
            st.pyplot(fig)
            
    #  Run Monte Carlo and plot the graph for every selected ticker
    for tick in tickers:
        MCS = MonteCarlo(tick=tick, data_source='yahoo',
                       time_horizon=Days, n_simulation=Times, seed=123)
        MCS.run_simulation()
        MCS.plot_simulation_price()
        
#==============================================================================
# Analysis
#==============================================================================
      
def tab5():
    
    # Dashboard title and data description
    st.title("Analysis")
    st.write("Data source: Yahoo Finance. URL: https://finance.yahoo.com/")
    
    # Loop over the selected tickers and i for colors
    for i, tick in enumerate(tickers):
       tick1 = yf.Ticker(tick)
       
       # Create actions DataFrame with Date, Dividends and Stock Splits columns
       actions = tick1.actions.reset_index()
       # Simplify the Date
       actions['Date'] = pd.to_datetime(actions['Date']).dt.date
       # Replace 0 values by nan to plot only the important stock splits
       actions['Stock Splits'].replace(0, np.nan, inplace=True)
       
       # Initiate the plot
       fig = go.Figure()
       
       # Create two y axis for 1 x axis
       fig = make_subplots(specs=[[{"secondary_y": True}]], 
                           shared_xaxes=True)
       
       # Create Line chart for Dividends on prymary y axis
       fig.add_trace(go.Scatter(x=actions['Date'], 
                                y=actions['Dividends'],
                                marker=dict(color=fig.layout['template']['layout']['colorway'][i]),
                                name = tick), 
                     secondary_y = False)
       
     # Create strip plot for Stock Splits on secondary y axis
       fig.add_trace(go.Scatter(x=actions['Date'], 
                          y=actions['Stock Splits'],
                          marker=dict(color=fig.layout['template']['layout']['colorway'][i +6]),
                          name = tick,
                          mode="markers"),
                          secondary_y = True)
       
       # Update both traces layout
       fig.update_layout( 
         title={'text': str(tick) + " Stock Dividends and Splits over time",
                'y':0.9,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'},
         hovermode="x unified",
         xaxis={'title':'Date'},
         yaxis1 = {'title':'Dividends'},
         yaxis2 = {'title':'Stock Splits'})
       fig.layout.yaxis2.showgrid=False
       
       # Display Chart with the traces 
       st.plotly_chart(fig, use_container_width=True)
       
       # Create an Expender to have a better overview of the data if necessary
       actions = actions.set_index('Date')
       with st.expander("Click here to get " + str(tick) + "'s Data"):
           st.table(actions)

       
#==============================================================================
# Main body
#==============================================================================

def run():
    
    st.set_page_config(
    page_title="Financial App",
    layout="wide",
    initial_sidebar_state="expanded")
    
    #Yahoo Finance Image
    image = Image.open('./img/yfinance.png')
    st.sidebar.image(image)
    
    # Get the list of stock tickers from S&P500
    ticker_list = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol']
    
    # Add selection box
    global tickers
    tickers = st.sidebar.multiselect("Select Ticker(s)", ticker_list)
    
    # Add update button
    Update = st.sidebar.button('Update')
    
    if Update:
        st.experimental_rerun()
      
    # Add a radio box
    select_tab = st.sidebar.radio("Select tab", ['Summary', 'Chart', 'Financial', 'Monte Carlo', 'Analysis'])
    
    # Show the selected tab
    if select_tab == 'Summary':
        tab1()
        
    elif select_tab == 'Chart':
        tab2()
        
    elif select_tab == 'Financial':
        tab3()
        
    elif select_tab == 'Monte Carlo':
        tab4()
        
    elif select_tab == 'Analysis':
        tab5()
    
if __name__ == "__main__":
    run()

###############################################################################
# END
###############################################################################