import streamlit as st
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import yfinance as yf
import pandas as pd


class Webapp:

    def __init__(self, heading, company, start_date=datetime.date.today(), end_date=datetime.date.today()) -> None:
        self.heading = heading
        self.data = self.load_data()
        self.end_date = start_date
        self.start_date = self.get_Start_Date()
        self.company = company

    def load_data(self):
        df = pd.read_csv('stock_data.csv')
        df['Text'] = df['Text'].astype(str)
        df['Sentiment'] = df['Sentiment'].astype(np.int64)
        daily_data = pd.read_csv('./raw_news/dailynews.csv')
        daily_data['Text'] = daily_data['Text'].astype(str)
        data = pd.concat([df, daily_data], ignore_index=True)
        data.drop_duplicates(subset="Text",
                             keep="first", inplace=True)
        data['Date'] = pd.to_datetime(data['Date']).dt.date
        data = data.sort_values('Date', ascending=True)
        return data

    def get_Start_Date(self):
        # today = datetime.date.today()
        return self.end_date - datetime.timedelta(days=self.end_date.weekday()+1)

    # def get_Start_Date(self):
    #     return self.end_date - datetime.timedelta(6)

    def plot_week_data(self):

        # Set the figure size
        plt.figure(figsize=(10, 6))
        week_data = self.data[(self.data['Date'] >= self.start_date) & (
            self.data['Date'] <= self.end_date)]

        sns.countplot(data=week_data, x="Date",
                      hue="Sentiment", palette="Paired")

        # Set the x-axis labels to be rotated for better visibility
        plt.xticks(rotation=45)

        # Set the x and y-axis labels
        plt.xlabel('Date')
        plt.ylabel('Count')

        # Set the title of the plot
        plt.title('Sentiment Distribution for a Week')

        # Show the legend
        plt.legend(('Neutral', 'Positive', 'Negative'))

        # Display the plot
        st.pyplot(plt)

    def plot_stock_data(self):

        # Retrieve the historical data
        print(self.start_date, " ", self.end_date)
        df = yf.download(
            self.company, start=self.start_date, end=self.end_date)

        fig = go.Figure(data=[go.Candlestick(x=df.index.values,
                        open=df['Open'],
                        high=df['High'],
                        low=df['Low'],
                        close=df['Close'])])

        fig.update_layout(
            width=1000,  # Set the width of the figure to 800 pixels
            height=500,  # Set the height of the figure to 400 pixels
            margin=dict(l=50, r=50, t=50, b=50),  # Set the margins of the plot
        )

        st.plotly_chart(fig, use_container_width=True)

        # fig.show().save()


if __name__ == "__main__":
    st.title('Stock Analysis website')

    start_date = st.date_input("Enter the start date")
    # end_date = st.date_input("Enter the end date")

    # end_date = st.date_input("Enter a date", width=100)
    web = Webapp('Stock Analysis website', "hello",
                 start_date)
    web.plot_week_data()
    st.markdown('Stock prices')

    options = ["ICICI Bank", "Adani Enterprices", "SBI Bank", "HDFC Bank"]
    company = ['ICICIBANK.NS', 'ADANIENT.NS', 'SBIN.NS', 'HDFCBANK.NS']
    web.company = company[0]
    selected_option = st.selectbox("Select the company to show data", options)
    selected_company_symbol = company[options.index(selected_option)]
    # Display the selected option
    st.write("You selected:", selected_option)
    if datetime.datetime.today().weekday() < 5:
        web.plot_stock_data()
    else:
        st.write("Today is weekend, Stock market is closed!!!! ")
