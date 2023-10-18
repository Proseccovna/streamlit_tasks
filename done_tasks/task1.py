#Сделайте своё приложение на русском языке на платформе Streamlit с данными о котировках компании Apple c помощью библиотеки yfinance
import yfinance as yf
import streamlit as st

st.write("""
# **Данные о котировках компании Apple**

*C помощью библиотеки yfinance*

""")
tickerSymbol = 'AAPL'
#get data on this ticker
tickerData = yf.Ticker(tickerSymbol)
#get the historical prices for this ticker
tickerDf = tickerData.history(period='1d', start='2011-5-31', end='2023-5-31')
# Open	High	Low	Close	Volume	Dividends	Stock Splits
st.text(""" 
График №1
цена за акцию, $
""")
st.line_chart(tickerDf.Close)
st.text(""" 
График №2
Объем акций
""")
st.line_chart(tickerDf.Volume)
