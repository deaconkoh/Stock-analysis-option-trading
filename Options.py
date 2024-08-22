import yfinance as yf
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from bs4 import BeautifulSoup
import requests
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from datetime import datetime
from scipy.signal import argrelextrema
from matplotlib import pyplot as plt 
from sklearn.neighbors import KernelDensity
from scipy.signal import find_peaks
from sklearn.model_selection import GridSearchCV

#Retriving stock historical data from yfinance and information from user
symbol = input("Enter the Ticker symbol: ")
hist = yf.Ticker(symbol).history(auto_adjust=False, actions=False, period="1y")
df = pd.DataFrame(hist)
three_mth_data = df.tail(64)

def volume_score(): #Calculating the relative volume of the stock to determine if the volume is greater than usual
    current_volume = three_mth_data["Volume"].iloc[-1]
    avg_vol = three_mth_data["Volume"].mean()
    rel_vol = current_volume/avg_vol
    return rel_vol #if relative volume is greater than or equal to 2 means increased interest

def cal_beta(): #calculating the Beta & Standard deviation of the stock to determine it's volatility
    #Creating a temporary dataframe with S&P500 adjusted close
    market_hist = yf.Ticker("SPY").history(auto_adjust=False, actions=False, period="1y")
    market_return = market_hist["Adj Close"].pct_change().dropna()
    stock_return = three_mth_data["Adj Close"].pct_change().dropna()
    temp_df = pd.DataFrame({"Stock": stock_return, "Market": market_return}).dropna()
    
    #calculating the beta
    cov_matrix = temp_df.cov().iloc[0,1]
    var_matrix = temp_df["Market"].var()
    beta = cov_matrix/var_matrix
    
    #calculating the standard deviation of the stock price
    daily_sd = stock_return.std()
    annual_sd = daily_sd*np.sqrt(252)
    
    return beta, annual_sd

def trend(): #Using Golden/Death cross to determine if the stock is bullish or bearish
    if df.shape[0] < 200:
        return "Insufficient data"
    df["SMA 50"] = df["Adj Close"].rolling(window=50).mean()
    df["SMA 200"] = df["Adj Close"].rolling(window=200).mean()
    if df["SMA 50"].iloc[-1] > df["SMA 200"].iloc[-1]:
        return "Bullish"
    elif df["SMA 50"].iloc[-1] < df["SMA 200"].iloc[-1]:
        return "Bearish"
    else:
        return "No clear trend"
    
def relative_strength():#determining the RSI of the stock
    rsi = RSIIndicator(close=df["Adj Close"], window=14)
    df["RSI"] = rsi.rsi()
    if df["RSI"].iloc[-1] >= 70:
        return "Overbought"
    elif df["RSI"].iloc[-1] <=30:
        return "Oversold" 
    else:
        return "Neutral"

def cal_macd(): #Calculate the MACD to support the results from Golden/death cross
    df["MACD"] = df["Adj Close"].ewm(span=12, adjust=False).mean() - df["Adj Close"].ewm(span=26, adjust=False).mean()
    df["Signal Line"] = df["MACD"].ewm(span=9, adjust=False).mean()
    
    if df["MACD"].iloc[-1] > df["Signal Line"].iloc[-1]:
        return "Bullish"
    else:
        return "Bearish"
    
def news_sentiment():#Webscrape from finviz to determine the sentiment of news for the stock
    try:
        url = "https://finviz.com/quote.ashx?t=" + symbol
        response = requests.get(url=url, headers={"user-agent": "my-app"})
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print("Failed to fetch news data: ", e)
        return None
    
    soup = BeautifulSoup(response.text, "html.parser")
    
    parsed_data = []

    news_table = soup.find(id="news-table")
    if news_table is None:
        print("News table not found")
        return None
    
    for row in news_table.findAll("tr"):
        title = row.a.text
        timestamp = row.td.text.strip().split(" ")
        
        if len(timestamp)==1:
            time = timestamp[0]
        else:
            date = timestamp[0]
            time = timestamp[1]
        parsed_data.append([date, time, title])

    sentiment_df = pd.DataFrame(parsed_data, columns=["Date", "Time", "Title"])
    vadar = SentimentIntensityAnalyzer()

    f = lambda title: vadar.polarity_scores(title)["compound"]
    sentiment_df["Compound"] = sentiment_df["Title"].apply(f)

    current_date = datetime.now().strftime('%b-%d-%y')
    sentiment_df["Date"] = sentiment_df["Date"].replace("Today", current_date)
    sentiment_df["Date"] = pd.to_datetime(sentiment_df['Date'], format="%b-%d-%y").dt.date
    mean_df = sentiment_df.groupby(["Date"]).mean(numeric_only=True)
    today_sentiment = mean_df.iloc[-1]
    
    if today_sentiment["Compound"] > 0:
        return "Positive"
    elif today_sentiment["Compound"]<0:
        return "Negative"
    else:
        return "Neutral"

def supoort_resistance(): #Created a program that finds the stock supoort & resistance by grabbing heavily densed turning points, 
    raw_data = yf.Ticker(symbol).history(auto_adjust=False, actions=False, interval="5m")
    sr_df = pd.DataFrame(raw_data)
    sr_df.reset_index(inplace=True) #converting the datetime index as a column rather than a index
    sr_df["Datetime"] = pd.to_datetime(sr_df["Datetime"], unit="ms")

    sample = sr_df["Close"].to_numpy().flatten() #creating a numpy array

    #grab turning points ie. max & min
    maxima = argrelextrema(sample, np.greater)
    minima = argrelextrema(sample, np.less)
    extrema = np.concatenate((maxima, minima),axis=1)[0]
    extrema_price = np.concatenate((sample[maxima], sample[minima]))

    initial_price = extrema_price[0]

    bandwidth = np.linspace(0.01,1.0,50)
    grid = GridSearchCV(KernelDensity(kernel="gaussian"),{"bandwidth":bandwidth})
    grid.fit(extrema_price.reshape(-1,1))
    best_bandwidth = grid.best_params_["bandwidth"]

    kde = KernelDensity(kernel="gaussian", bandwidth=best_bandwidth).fit(extrema_price.reshape(-1,1))

    a, b = min(extrema_price), max(extrema_price)
    price_range = np.linspace(a,b,4000).reshape(-1,1)
    pdf = np.exp(kde.score_samples(price_range))
    peaks = find_peaks(pdf)[0]
    return price_range[peaks]

def main():
    #check for volatility
    volatility = False
    beta,std = cal_beta()
    if beta > 1 or std >= 0.20:
        volatility = True
    else:
        print(f"the stock is not volatile enough, it has a Beta of {round(beta,2)} and Standard deviation of {round(std,2)}")
            
    #check for relative volume
    rel_vol = False
    if volume_score()> 1:
        rel_vol = True
    else:
        print(f"the relative trading volume for {symbol} is too low, it has a trading relative trading volume of {round(volume_score(),2)}")
    
      
    #check if the closing price is near support/resistance
    suitable = False
    sr_levels = supoort_resistance()
    near_levels = ({"Support":[], "Resistance":[]})
    closing_price = df["Adj Close"].iloc[-1]
    margin_of_error = closing_price*0.01
    for sr in sr_levels:
        if abs(closing_price - sr) <= margin_of_error:
            if closing_price > sr:
                near_levels["Support"].append(sr)
            else:
                near_levels["Resistance"].append(sr)
    if len(near_levels["Resistance"]) > 0:
        print(f"The nearest resistance is {max(near_levels["Resistance"])}")
        suitable = True
    elif len(near_levels["Support"]) >0:
        print(f"The nearest support is {min(near_levels["Support"])}")
        suitable = True
    else:
        print(f"The stock is not near the support or resistance, Support: {min(near_levels["Support"])} & Resistance: {max(near_levels["Resistance"])}")
    
    updown = ""
    if trend() == "Bullish" and cal_macd()=="Bullish":
            updown = "Bullish"
    elif trend() == "Bearish" and cal_macd()=="Bearish":
        updown = "Bearish"
    else:
        updown = "No clear trend"

    print(f"News sentiment: {news_sentiment()}") 
    print(f"Trend: {updown}")
    print(f"RSI: {relative_strength()}")
    
    if rel_vol == False or volatility == False or suitable == False:
        print("This stock is not suitable for option trading")

if __name__ == "__main__":
    main()
