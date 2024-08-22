# Stock Analysis for Options Trading

## Overview

This Python script provides a comprehensive analysis of a given stock to determine its suitability for options trading. The script evaluates the stock's volatility, relative trading volume, and proximity to support or resistance levels. Additionally, it assesses the stock's news sentiment, trend, and Relative Strength Index (RSI) to provide a complete picture of its trading potential.

## Features

- **Volatility Analysis:** Determines if the stock is volatile enough based on its beta and standard deviation.
- **Relative Volume Calculation:** Assesses whether the stock's trading volume is higher than usual.
- **Support and Resistance Levels:** Identifies if the stock is near any key support or resistance levels.
- **Trend Detection:** Uses Golden/Death Cross and MACD to determine the stock's trend.
- **RSI Calculation:** Provides the RSI to indicate if the stock is overbought or oversold.
- **News Sentiment Analysis:** Scrapes and analyzes recent news sentiment related to the stock.

## Requirements

- `yfinance` - For fetching historical stock data.
- `pandas` - For data manipulation and analysis.
- `numpy` - For numerical operations.
- `ta` - For calculating the RSI indicator.
- `beautifulsoup4` - For web scraping news data.
- `requests` - For making HTTP requests.
- `nltk` - For sentiment analysis.
- `scipy` - For finding peaks and other signal processing tasks.
- `matplotlib` - For plotting (optional, if you include plots in your analysis).
- `sklearn` - For machine learning tasks, like GridSearchCV and KernelDensity.

## Usage

### Run the Script

Execute the script in your Python environment.

### Input

When prompted, enter the ticker symbol of the stock you want to analyze.

### Output

The script will output:

- **Volatility Check:** Indicates if the stock is sufficiently volatile.
- **Relative Volume:** Shows if the trading volume is higher than usual.
- **Support/Resistance Levels:** Provides the nearest support or resistance level.
- **News Sentiment:** Displays the sentiment of recent news about the stock.
- **Trend:** Indicates if the stock is bullish, bearish, or has no clear trend.
- **RSI:** Shows if the stock is overbought, oversold, or neutral.

### Example

```bash
Enter the Ticker symbol: AAPL
The stock is volatile & has high relative trading volume, and is near resistance.
The nearest resistance is 178.25
News sentiment: Positive
Trend: Bullish
RSI: Neutral

```

### Notes

- The script uses a 1-year historical data period for analysis.
- Ensure that your internet connection is active for fetching data and scraping news.
- Adjustments may be necessary depending on data availability and API changes.
