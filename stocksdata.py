import psycopg2
import pandas as pd
from bs4 import BeautifulSoup
import requests
from datetime import datetime,date,timedelta
import threading
import logging
import sys
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("debug.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

DATABASE = "stocks"
USER = "postgres"
PASSWORD="kingsuk"
HOST="127.0.0.1"
PORT='5432'

URL1  = "https://economictimes.indiatimes.com/markets/stocks/news/articlelist/msid-2146843,"
URL2 = "https://www.livemint.com/market/stock-market-news/"
URL3 = "https://www.moneycontrol.com/news/business/markets/"
URL4 = "https://www.business-standard.com/category/markets-news-1060101.htm/"
class Database:
    
    def __init__(self) -> None:
        self.conn = self.connection()
        self.cursor = self.conn.cursor()

    def connection(self):
        return psycopg2.connect(
   database=DATABASE, user=USER, password=PASSWORD, host=HOST, port=PORT)
    
    def fetch_data(self,query):
        self.cursor.execute(query)
        result= self.cursor.fetchall()
        return result
    
    def insert_data(self,query):
        self.cursor.execute(query)
        self.conn.commit()

today = date.today() - timedelta(1)
today = str(today) + ' 11:55 PM'
stop_date = datetime.strptime(today, '%Y-%m-%d %I:%M %p')
mcnews =[]
mctimes = []
mnews =[]
mtimes = []
bsnews =[]
bstimes = []
ecnews =[]
ectimes = []

class News:


    def __init__(self,url,element,classname):
        self.url = url
        self.element = element
        self.classname=classname  

    def fetch_from_url(self):
        try:
            page = requests.get(self.url)
            soup = BeautifulSoup(page.content, "html.parser")
            result = soup.find_all(self.element, class_=self.classname)
            logging.info("Extacted necessary data from soup ")
            return result
        except Exception as e:
            logging.warning("Exception caught at fetching url")
            logging.error(e)
        return None
        


    def economic_news(self)-> bool:
        for x in range(1,4):
            self.url = self.url + "page-" + str(x) + ".cms"      
            result = self.fetch_from_url()
            logging.info("Data fetched from economic news")
            for i in result:    
                data = i.find('a')
                times = i.find('time')
                times = times.text
                logging.info("news and time extracted")
                stop_time = datetime.strptime(
                    str(times).strip(), '%b %d, %Y, %I:%M %p IST')
                logging.info("Time formated")
                data = data.text.strip()
                logging.info("Data formated")
                if stop_time <= stop_date:
                    logging.info("Fetching end at {} where given times was {}".format(stop_time,stop_date))
                    return
                # economic_times_data[data]=stop_time
                
                ecnews.append(data)
                logging.info("Data appended to list")
                ectimes.append(stop_time)
                logging.info("Time appended to list")
            


    def mint_news(self) -> bool:
        for x in range(1,4):
            self.url = self.url + "page-" + str(x)
            result = self.fetch_from_url()
            logging.info("Data fetched from mint news")
            for i in result:
                data = i.find('a')
                tim = i.find('span')
                logging.info("news and time extracted")
                tim = "".join(
                        [str(x)[136:161] for x in i.contents if "<span data-expandedtime" in str(x)])      
                stop_time = datetime.strptime(str(tim), '%d %b %Y, %I:%M %p IST')
                logging.info("Time formated")
                data = data.text.strip()
                logging.info("Data formated")
                if stop_time <= stop_date:
                    logging.info("Fetching end at {} where given times was {}".format(stop_time,stop_date))
                    return
                # mint_data[data] = stop_time
                mnews.append(data)
                logging.info("Data appended to list")
                mtimes.append(stop_time)
                logging.info("Time appended to list")
                
    def money_control_news(self):
        for x in range(1,4):
            self.url = self.url +"page-" + str(x)
            result = self.fetch_from_url()
            logging.info("Data fetched from money control news")
            for i in result:
                stop_time = ''
                page_text = (i.get_text()).strip()
                page_text = str(page_text).replace(
                        i.find('p').text.strip(), "").strip()

                page_text = page_text.split('IST')
                logging.info("news and time extracted")
                stop_time = datetime.strptime(
                        str(page_text[0]).strip(), '%B %d, %Y %I:%M %p')
                logging.info("Time formated")
                logging.info("Data formated")
                if stop_time <= stop_date:
                    logging.info("Fetching end at {} where given times was {}".format(stop_time,stop_date))
                    return
                # money_control_data[(page_text[1])] = stop_time
                mcnews.append(page_text[1])
                logging.info("Data appended to list")
                mctimes.append(stop_time)
                logging.info("Time appended to list")


    def business_standard_news(self):
        date_stop = datetime.now().date()
        for x in range(1,4):
            self.url = self.url + str(x)
            result = self.fetch_from_url()
            logging.info("Data fetched from business standard news")
            for i in result:
                stop_time = ''
                page_text = (i.get_text()).strip()
                logging.info("news and time extracted")
                page_text = [x.strip("\n") for index, x in enumerate(
                    list(page_text.splitlines(True))) if index < 2]
                logging.info("Data formated")
                stop_time = datetime.strptime(str(page_text[0]), '%B %d, %Y, %A').date()
                logging.info("Time formated")
                if stop_time < date_stop:
                    logging.info("Fetching end at {} where given times was {}".format(stop_time,stop_date))
                    return
                # business_standard_data[(page_text[1])] = stop_time
                bsnews.append(page_text[1])
                logging.info("Data appended to list")
                bstimes.append(stop_time)
                logging.info("Time appended to list")



if __name__ == "__main__":
    ec = News(URL1,'div','eachStory')
    logging.info("Economic times object created")
    mi = News(URL2,'div','headlineSec')
    logging.info("Mint object created")
    mc = News(URL3,'li','clearfix')
    logging.info('Money control object created')
    bs = News(URL4,'div','listing-txt')
    logging.info("Business Standard object created")
    economic_times = threading.Thread(target=ec.economic_news, name="economic_times")
    logging.info("Economic times thread created")
    mint = threading.Thread(target=mi.mint_news, name="mint")
    logging.info("Mint thread created")
    money_control = threading.Thread(target=mc.money_control_news, name="moneycontrol")
    logging.info("Money control thread created")
    business_standard = threading.Thread(target=bs.business_standard_news, name="business_standard")
    logging.info("Business Standard thread created")
    economic_times.start()
    logging.info("Economic times thread started")
    mint.start()
    logging.info("Mint thread started")
    money_control.start()
    logging.info("money control thread started")    
    business_standard.start()
    logging.info("business standard thread started")
  

    money_control.join()
    mint.join()
    business_standard.join()
    economic_times.join()
    logging.info("All thread completed")
    news = mcnews+mnews+ecnews+bsnews
    times = mctimes+mtimes+ectimes+bstimes
    data = pd.DataFrame({'Headline':news,'Date':times})
    data.to_csv('./raw_news/NewsData-' +
                      str(date.today()) + '.csv')
    logging.info("csv created")
    
    # db = Database()
    # df = pd.read_csv('stock_data.csv')
    # df['Text']=df['Text'].str.replace("\'","\\")
    # for ind in df.index:
    #     query = "INSERT INTO NEWS(news, news_date, sentiment) VALUES ('" + df['Text'][ind] +"','" + df['Date'][ind]+"','"+str(df['Sentiment'][ind])+"')"
    #     db.insert_data(query)
    # print(db.fetch_data('''SELECT * from NEWS'''))
    # db.conn.close()

        



























