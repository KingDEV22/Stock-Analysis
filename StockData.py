# +
import pandas as pd
import numpy as np
import git
import subprocess
import datetime
from bs4 import BeautifulSoup
import requests
import os
from datetime import date, timedelta
import datetime
import pandas as pd
import numpy as np
import warnings
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords


def fxn():
    warnings.warn("deprecated", DeprecationWarning)


with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()


url = ["https://www.livemint.com/market/stock-market-news/", "https://economictimes.indiatimes.com/markets/stocks/news/articlelist/msid-2146843,",
       "https://www.moneycontrol.com/news/business/markets/"]

today = date.today() - timedelta(1)
stop_dates = today.strftime('%d')
stop_time = datetime.time(9, 0, 0)
news = []
time = []


def get_news_url(url):
    page = requests.get(url)
    soup = BeautifulSoup(page.content, "html.parser")
    return soup


def extact_data(soup, ele, classname, ele1, ele2, type):
    result = soup.find_all(ele, class_=classname)
    for i in result:
        data = i.find(ele1)
        times = i.find(ele2)
        if type == 'mint':
            data = data.text.strip()
            times = "".join(
                [str(x)[136:161] for x in i.contents if "<span data-expandedtime" in str(x)])
        elif type == 'et':
            times = times.text
            data = data.text.strip()
        else:
            data = data.get('title')
            times = times.text
        *get_dat, get_tim = times.split(',')
        if type == 'mc':
            hour, sec = (get_tim.strip().split(' ')[1].split(":"))
            noon = datetime.time(int(hour), int(sec), 0)
        else:
            hour, sec = (get_tim.strip().split(' ')[0].split(":"))
            noon = datetime.time(int(hour), int(sec), 0)

        if stop_dates in get_dat[0] and stop_time >= noon:
            return False

        news.append(data)
        time.append(times)


def news_return(data1, data2):

    # Empty data frame
    contain_values1 = pd.DataFrame()
    data1['Headline'] = data1['Headline'].str.replace('[^\w\s]', ' ')
    data1['Headline'] = " "+data1['Headline']+" "

    # iterate all company names to find out the match
    for i in range(len(data2)-1):
        try:
            keyword = data2["Symbol"][i]
            CompanyName = data2["Company Name"][i]

            keyword = " "+keyword+" "
            CompanyName = " "+CompanyName+" "

            contain_values = data1[(data1['Headline'].str.contains(keyword)) |
                                   (data1['Headline'].str.contains(CompanyName))]
            if(keyword == 'ttl'):
                continue

            contain_values['Symbol'] = data2["Symbol"][i]
            contain_values['Company Name'] = data2["Company Name"][i]

            if(i % 1000 == 0):
                print(i, "Complited")
            contain_values1 = contain_values1.append(
                contain_values, ignore_index=True)

        except:
            print("Company name is small at index ", i)

    return contain_values1


def __stopwordsRemoval__(word, stopWords):
    words = word_tokenize(word)
    wordsFiltered = ""

    for w in words:
        if w not in stopWords:
            wordsFiltered = wordsFiltered+w+" "
    return wordsFiltered[:len(wordsFiltered)-1]


def __fetchNews__(url):
    for i in range(3):
        for j in range(1, 4):
            if i == 1:
                path = url[i] + "page-" + str(j) + ".cms"
            else:
                path = url[i] + "page-" + str(j)
            page = get_news_url(path)
            if i == 0:
                if not extact_data(page, 'div', 'headlineSec', 'a', 'span', 'mint'):
                    break
            if i == 1:
                if not extact_data(page, 'div', 'eachStory', 'a', 'time', 'et'):
                    break
            else:
                if not extact_data(page, 'li', 'clearfix', 'a', 'span', 'mc'):
                    break

    stock_data = pd.DataFrame({'Headline': news,
                               'Date': time})
    stock_data.to_csv('.\\raw_news\\NewsData-' +
                      str(datetime.date.today()) + '.csv')


def __NewsToStock__():
    data1 = pd.read_csv('./raw_news/NewsData-' +
                        str(datetime.date.today()) + '.csv')
    data2 = pd.read_csv('MCAP31122022_0.csv')
    data1 = data1[{'Headline', 'Date'}]
    data2 = data2[{"Symbol", "Company Name"}]

    data1['Headline'] = data1['Headline'].apply(str.lower)
    data2['Symbol'] = data2['Symbol'].apply(str)
    data2['Symbol'] = data2['Symbol'].apply(str.lower)
    data2['Company Name'] = data2['Company Name'].apply(str)
    data2['Company Name'] = data2['Company Name'].apply(str.lower)
    data2 = data2.iloc[0:len(data2)-1, :]

    # List of words not needed
    stopWords = set(stopwords.words("english"))
    stopWords.add("limited")
    stopWords.add("( india )")
    stopWords.add("(")
    stopWords.add(")")
    stopWords.remove("of")

    data2["Company Name"] = data2.apply(
        lambda x: __stopwordsRemoval__(x["Company Name"], stopWords), axis=1)

    data3 = news_return(data1, data2)
    data3.to_csv('.\\news\\StockData_' + str(datetime.date.today()) + '.csv')


# Driver Code
__fetchNews__(url)
__NewsToStock__()
repo = git.Repo('.')
subprocess.check_output("git add .", stderr=subprocess.PIPE)
repo.index.commit('news fetch for ' + str(datetime.date.today()))
repo.remotes.origin.push()
# -
