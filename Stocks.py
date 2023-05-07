import threading
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd
from bs4 import BeautifulSoup
import requests
import pickle
import os
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


mcnews = []
mctimes = []
mnews = []
mtimes = []
bsnews = []
bstimes = []
ecnews = []
ectimes = []


today = date.today() - timedelta(1)
today = str(today) + ' 09:00 PM'
stop_date = datetime.strptime(today, '%Y-%m-%d %I:%M %p')


def fetch_from_url(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.83 Safari/537.36'}
    page = requests.get(url, headers=headers)
    soup = BeautifulSoup(page.content, "html.parser")
    logging.info("fetch from " + url)
    return soup


def extract_data_from_page(page, ele, classname):
    return page.find_all(ele, class_=classname)


def economic_news():
    today = datetime.now()
    month = today.strftime("%b")
    logging.info("fetching from economic times")
    for x in range(1, 4):
        url = "https://economictimes.indiatimes.com/markets/stocks/news/articlelist/msid-2146843,"
        page = fetch_from_url(url + "page-" + str(x) + ".cms")
        result = extract_data_from_page(page, 'div', 'eachStory')
        for i in result:
            data = i.find('a')
            times = i.find('time')
            times = times.text
            stop_time = datetime.strptime(
                str(times).strip(), '%b %d, %Y, %I:%M %p IST')
            data = data.text.strip()
            if stop_time <= stop_date:
                logging.info("fetching comepleted from economic times")
                return

            ecnews.append(data)
            ectimes.append(format_date(stop_time))


def mint_news():
    logging.info("fetching from mint")
    for x in range(1, 4):
        url = "https://www.livemint.com/market/stock-market-news/"
        page = fetch_from_url(url+"page-" + str(x))

        result = extract_data_from_page(page, 'div', 'headlineSec')
        for i in result:
            data = i.find('a')
            tim = i.find('span')

            data = data.text.strip()
            tim = "".join(
                [str(x)[136:161] for x in i.contents if "<span data-expandedtime" in str(x)])

            stop_time = datetime.strptime(str(tim), '%d %b %Y, %I:%M %p IST')
            if stop_time <= stop_date:
                logging.info("fetching completed from mint")
                return
            mnews.append(data)
            mtimes.append(format_date(stop_time))


def money_control_news():
    logging.info("fetching from money control")
    for x in range(1, 4):
        url = "https://www.moneycontrol.com/news/business/markets/"
        page = fetch_from_url(url+"page-" + str(x))

        result = extract_data_from_page(page, 'li', 'clearfix')
        for i in result:
            stop_time = ''
            page_text = (i.get_text()).strip()
            page_text = str(page_text).replace(
                i.find('p').text.strip(), "").strip()

            page_text = page_text.split('IST')
            stop_time = datetime.strptime(
                str(page_text[0]).strip(), '%B %d, %Y %I:%M %p')
            if stop_time <= stop_date:
                logging.info("fetching completed from money control")
                return
            mcnews.append(page_text[1])
            mctimes.append(format_date(stop_time))


def business_standard_news():
    date_stop = datetime.now().date()
    logging.info("fetching from business standard")
    url = "https://www.business-standard.com/markets/news"
    page = fetch_from_url(url)
    result = extract_data_from_page(page, 'div', 'cardlist')
    for i in result:
        data = i.find('a')
        bsnews.append(data.text.strip())
        bstimes.append(date_stop)
    logging.info("fetching completed from business standard")


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
    logging.info("stopwords removed and lematization completed")
    return data


def ml_model(data):
    model = pickle.load(open('./pkl/model.pkl', 'rb'))
    model1 = pickle.load(open('./pkl/model1.pkl', 'rb'))
    tfidf = pickle.load(open('./pkl/vectorizer.pkl', 'rb'))
    logging.info("model loaded")
    x_test = tfidf.transform(data['Tok_text'])
    y_pred = model.predict(x_test)
    logging.info("data predicted")
    data['sdg_pred'] = y_pred
    y_pred1 = model1.predict(x_test)
    logging.info("data predicted")
    data['lr_pred'] = y_pred1
    return data


def dp_model(data):
    tokenizer = pickle.load(open('./pkl/tokenizer.pkl', 'rb'))
    max_length = pickle.load(open('./pkl/max_length.pkl', 'rb'))
    sequences = tokenizer.texts_to_sequences(data['Tok_text'])
    padded_sequences = pad_sequences(sequences, maxlen=max_length)
    lstm_model = load_model('./pkl/Lstm.h5')
    logging.info("lstm model loaded")
    lstm_pred = lstm_model.predict(padded_sequences)
    data['lstm_pred'] = np.argmax(lstm_pred, axis=1)
    logging.info("data predicted")
    biGru_model = load_model('./pkl/biGRU.h5')
    logging.info("GRU model loaded")
    biGru_pred = biGru_model.predict(padded_sequences)
    data['biGru_pred'] = np.argmax(biGru_pred, axis=1)
    logging.info("data predicted")
    biGru_model1 = load_model('./pkl/biGRU1.h5')
    logging.info("GRU model 1 loaded")
    biGru_pred1 = biGru_model1.predict(padded_sequences)
    data['biGru_pred1'] = np.argmax(biGru_pred1, axis=1)
    logging.info("data predicted")
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


if __name__ == "__main__":
    money_control = threading.Thread(
        target=money_control_news, name="moneycontrol")
    mint = threading.Thread(target=mint_news, name="mint")
    business_standard = threading.Thread(
        target=business_standard_news, name="business_standard")
    economic_times = threading.Thread(
        target=economic_news, name="economic_times")

    money_control.start()
    mint.start()
    business_standard.start()
    economic_times.start()

    money_control.join()
    mint.join()
    business_standard.join()
    economic_times.join()
    news = mcnews+mnews+ecnews+bsnews
    times = mctimes+mtimes+ectimes+bstimes
    data = pd.DataFrame({'Text': news, 'Date': times})
    logging.info("dataframe created")
    data['Date'] = pd.to_datetime(data['Date']).dt.date
    data = filter_data(data)
    data = ml_model(data)
    data = dp_model(data)
    sentimet = data.apply(calculate_sentiment, axis=1)
    data = data.iloc[:, :2]
    data['Sentiment'] = sentimet
    logging.info("Sentiment calculated")
    handle_dataset(data)
