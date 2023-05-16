import threading
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd
from bs4 import BeautifulSoup
import requests
import pickle
import numpy as np
from datetime import datetime, date, timedelta
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set the logging level to INFO (or any desired level)
    # Set the log message format
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'  # Set the date and time format
)

# Global variables
money_control_news = []
money_control_datetimes = []
mint_news = []
mint_datetimes = []
business_standard_news = []
business_standard_datetimes = []
economic_times_news = []
economic_times_datetimes = []


today = date.today() - timedelta(1)
today = str(today) + ' 09:00 PM'
stop_date = datetime.strptime(today, '%Y-%m-%d %I:%M %p')


# loading comapany filter
check_set = pickle.load(open('./pkl/check_comapny.pkl', 'rb'))

# Methods declared


def fetch_from_url(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.83 Safari/537.36'}
    page = requests.get(url, headers=headers)
    soup = BeautifulSoup(page.content, "html.parser")
    logging.info("Fetch from " + url)
    return soup


def extract_data_from_page(page, ele, classname):
    return page.find_all(ele, class_=classname)


def economic_news_func():
    logging.info("Extracting data from economic times news")
    for page_no in range(1, 4):
        url = "https://economictimes.indiatimes.com/markets/stocks/news/articlelist/msid-2146843,"
        page = fetch_from_url(url + "page-" + str(page_no) + ".cms")
        results = extract_data_from_page(page, 'div', 'eachStory')
        for result in results:
            data = result.find('a')
            times = result.find('time')
            times = times.text
            stop_time = datetime.strptime(
                str(times).strip(), '%b %d, %Y, %I:%M %p IST')
            data = data.text.strip()
            if stop_time <= stop_date:
                logging.info("Extraction comepleted from economic times news")
                return

            economic_times_news.append(data)
            economic_times_datetimes.append(format_date(stop_time))


def mint_news_func():
    logging.info("Extracting data from mint news")
    for page_no in range(750, 890):
        url = "https://www.livemint.com/market/stock-market-news/"
        page = fetch_from_url(url+"page-" + str(page_no))

        results = extract_data_from_page(page, 'div', 'headlineSec')
        for result in results:
            data = result.find('a')
            tim = result.find('span')

            data = data.text.strip()
            tim = "".join(
                [str(x)[136:161] for x in result.contents if "<span data-expandedtime" in str(x)])

            stop_time = datetime.strptime(str(tim), '%d %b %Y, %I:%M %p IST')
            if stop_time <= stop_date:
                logging.info("Extraction completed from mint news")
                return
            mint_news.append(data)
            mint_datetimes.append(format_date(stop_time))


def money_control_news_func():
    logging.info("Extracting data from money control news")
    for page_no in range(1, 4):
        url = "https://www.moneycontrol.com/news/business/markets/"
        page = fetch_from_url(url+"page-" + str(page_no))

        results = extract_data_from_page(page, 'li', 'clearfix')
        for result in results:
            stop_time = ''
            page_text = (result.get_text()).strip()
            page_text = str(page_text).replace(
                result.find('p').text.strip(), "").strip()

            page_text = page_text.split('IST')
            stop_time = datetime.strptime(
                str(page_text[0]).strip(), '%B %d, %Y %I:%M %p')
            if stop_time <= stop_date:
                logging.info("Extraction completed from money control news")
                return
            money_control_news.append(page_text[1])
            money_control_datetimes.append(format_date(stop_time))


def business_standard_news_func():
    date_stop = datetime.now().date()
    logging.info("Extracting data from business standard news")
    url = "https://www.business-standard.com/markets/news"
    page = fetch_from_url(url)
    results = extract_data_from_page(page, 'div', 'cardlist')
    for result in results:
        data = result.find('a')
        business_standard_news.append(data.text.strip())
        business_standard_datetimes.append(date_stop)
    logging.info("Extraction completed from business standard news")


def text_data_cleaning(sentence):
    lemmatizer = WordNetLemmatizer()
    sent = preprocess_text(sentence)
    doc = nltk.word_tokenize(sent)
    lemma = [lemmatizer.lemmatize(word, pos="v") for word in doc]
    return append_message(lemma)


def preprocess_text(sen):
    to_remove = ['up', 'down', 'low', 'high', 'below', 'less', 'fall']
    new_stopwords = set(stopwords.words('english')).difference(to_remove)
    sentence = str(sen).lower()
    sentence = re.sub('[^a-zA-Z]', ' ', sentence)
    sentence = re.sub('rs|cr|crore|point|points|pt|stock', ' ', sentence)
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)
    sentence = re.sub(r'\s+', ' ', sentence)
    pattern = re.compile(r'\b(' + r'|'.join(new_stopwords) + r')\b\s*')
    sentence = pattern.sub('', sentence)
    return sentence


def append_message(text):
    str = " "
    return (str.join(text))


def filter_data(data):
    data = data[data['Text'].str.contains("\?") != True]
    data = data[data['Text'].str.contains("Wall Street|Amazon|Google") != True]
    data['Tok_text'] = data['Text'].apply(text_data_cleaning)
    logging.info("Stopwords removed and Lematization completed")
    return data


def ml_model(data):
    model = pickle.load(open('./pkl/model.pkl', 'rb'))
    model1 = pickle.load(open('./pkl/model1.pkl', 'rb'))
    tfidf = pickle.load(open('./pkl/vectorizer.pkl', 'rb'))
    logging.info("Model loaded")
    x_test = tfidf.transform(data['Tok_text'])
    y_pred = model.predict(x_test)
    logging.info("Data predicted")
    data['sdg_pred'] = y_pred
    y_pred1 = model1.predict(x_test)
    logging.info("Data predicted")
    data['lr_pred'] = y_pred1
    return data


def dp_model(data):
    tokenizer = pickle.load(open('./pkl/tokenizer.pkl', 'rb'))
    max_length = pickle.load(open('./pkl/max_length.pkl', 'rb'))
    sequences = tokenizer.texts_to_sequences(data['Tok_text'])
    padded_sequences = pad_sequences(sequences, maxlen=max_length)
    lstm_model = load_model('./pkl/Lstm.h5')
    logging.info("LSTM model loaded")
    lstm_pred = lstm_model.predict(padded_sequences)
    data['lstm_pred'] = np.argmax(lstm_pred, axis=1)
    logging.info("Data predicted")
    biGru_model = load_model('./pkl/biGRU.h5')
    logging.info("GRU model loaded")
    biGru_pred = biGru_model.predict(padded_sequences)
    data['biGru_pred'] = np.argmax(biGru_pred, axis=1)
    logging.info("Data predicted")
    biGru_model1 = load_model('./pkl/biGRU1.h5')
    logging.info("GRU model 1 loaded")
    biGru_pred1 = biGru_model1.predict(padded_sequences)
    data['biGru_pred1'] = np.argmax(biGru_pred1, axis=1)
    logging.info("Data predicted")
    return data


def format_date(timestamp):
    dt = datetime.strptime(str(timestamp), '%Y-%m-%d %H:%M:%S')
    formatted_date = dt.strftime('%Y-%m-%d')
    return formatted_date


def calculate_sentiment(data):
    sentiment = data.iloc[-3:].tolist()
    return max(sentiment, key=sentiment.count)


def handle_dataset(data):
    df1 = pd.read_csv('./raw_news/dailynews.csv')
    fg = pd.concat([df1, data], ignore_index=True)
    fg.drop_duplicates(subset="Text",
                       keep="first", inplace=True)
    fg.to_csv('./raw_news/dailynews.csv', index=False)
    logging.info("Dataset created")


def apply_company(text):
    text = text.replace('[^\w\s]', ' ')
    text = text.lower()
    text = set(text.split())
    for x in text:
        if x in check_set:
            return x
    return "N/A"


# Main function execution
if __name__ == "__main__":

    # Building the threads to get data from all four major news channels
    money_control = threading.Thread(
        target=money_control_news_func, name="moneycontrol")
    mint = threading.Thread(target=mint_news_func, name="mint")
    business_standard = threading.Thread(
        target=business_standard_news_func, name="business_standard")
    economic_times = threading.Thread(
        target=economic_news_func, name="economic_times")

    # Starting the Threads to get data from all four major news channels
    money_control.start()
    mint.start()
    business_standard.start()
    economic_times.start()

    # Joining the four threads so that furthur steps run after all the
    # data are being extracted first
    money_control.join()
    mint.join()
    business_standard.join()
    economic_times.join()

    # Starting main execution
    news = money_control_news + mint_news + \
        economic_times_news + business_standard_news
    times = money_control_datetimes + mint_datetimes + \
        economic_times_datetimes + business_standard_datetimes

    # Creating dataframes
    data = pd.DataFrame({'Text': news, 'Date': times})
    logging.info("dataframe created")
    data['Date'] = pd.to_datetime(data['Date']).dt.date
    data = filter_data(data)
    data = ml_model(data)
    data = dp_model(data)
    sentiment = data.apply(calculate_sentiment, axis=1)
    data = data.iloc[:, :2]
    data['Sentiment'] = sentiment
    logging.info("Sentiment calculated")
    data['Company'] = data['Text'].apply(apply_company)
    handle_dataset(data)
