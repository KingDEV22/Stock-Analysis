import threading
import time
import pandas as pd
from bs4 import BeautifulSoup
import requests
from datetime import datetime,date,timedelta
mcnews =[]
mctimes = []
mnews =[]
mtimes = []
bsnews =[]
bstimes = []
ecnews =[]
ectimes = []
# money_control_data ={'news': mcnews,'time':mctimes}
# mint_data = {'news': mnews,'time':mtimes}
# business_standard_data = {'news': bsnews,'time':bstimes}
# economic_times_data = {'news': ecnews,'time':ectimes}


today = date.today() - timedelta(1)
today = str(today) + ' 09:00 PM'
stop_date = datetime.strptime(today, '%Y-%m-%d %I:%M %p')

def fetch_from_url(url):
    page = requests.get(url)
    soup = BeautifulSoup(page.content, "html.parser")
    return soup

def extract_data_from_page(page,ele,classname):
     return page.find_all(ele, class_=classname)


def economic_news():
    today = datetime.now()
    month = today.strftime("%b")
    for x in range(1,4):
        url = "https://economictimes.indiatimes.com/markets/stocks/news/articlelist/msid-2146843,"
        page = fetch_from_url(url+ "page-" + str(x) + ".cms")        
        result = extract_data_from_page(page,'div', 'eachStory') 
        for i in result:    
            data = i.find('a')
            times = i.find('time')
            times = times.text
            stop_time = datetime.strptime(
                str(times).strip(), '%b %d, %Y, %I:%M %p IST')
            data = data.text.strip()
            if stop_time <= stop_date:
                return
            # economic_times_data[data]=stop_time
            ecnews.append(data)
            ectimes.append(stop_time)
           


def mint_news():
    for x in range(1,4):
        url = "https://www.livemint.com/market/stock-market-news/"
        page = fetch_from_url(url+"page-" + str(x))
        
        result = extract_data_from_page(page,'div','headlineSec') 
        for i in result:
            data = i.find('a')
            tim = i.find('span')
        
            data = data.text.strip()
            tim = "".join(
                    [str(x)[136:161] for x in i.contents if "<span data-expandedtime" in str(x)])
    
            stop_time = datetime.strptime(str(tim), '%d %b %Y, %I:%M %p IST')
            if stop_time <= stop_date:
                return
            # mint_data[data] = stop_time
            mnews.append(data)
            mtimes.append(stop_time)
            

        

def money_control_news():
    for x in range(1,4):
        url = "https://www.moneycontrol.com/news/business/markets/"
        page = fetch_from_url(url+"page-" + str(x))
        
        result = extract_data_from_page(page,'li', 'clearfix') 
        for i in result:
            stop_time = ''
            page_text = (i.get_text()).strip()
            page_text = str(page_text).replace(
                    i.find('p').text.strip(), "").strip()

            page_text = page_text.split('IST')
            stop_time = datetime.strptime(
                    str(page_text[0]).strip(), '%B %d, %Y %I:%M %p')
            if stop_time <= stop_date:
               return
            # money_control_data[(page_text[1])] = stop_time
            mcnews.append(page_text[1])
            mctimes.append(stop_time)


def business_standard_news():
    date_stop = datetime.now().date()
    for x in range(1,4):
        url = "https://www.business-standard.com/category/markets-news-1060101.htm/"
        page = fetch_from_url(url + str(x))
        result = extract_data_from_page(page,'div', 'listing-txt') 
        for i in result:
            stop_time = ''
            page_text = (i.get_text()).strip()
            page_text = [x.strip("\n") for index, x in enumerate(
                list(page_text.splitlines(True))) if index < 2]
      
            stop_time = datetime.strptime(str(page_text[0]), '%B %d, %Y, %A').date()
           
            if stop_time < date_stop:
               return
            # business_standard_data[(page_text[1])] = stop_time
            bsnews.append(page_text[1])
            bstimes.append(stop_time)


if __name__ == "__main__":
    money_control = threading.Thread(target=money_control_news, name="moneycontrol")
    mint = threading.Thread(target=mint_news, name="mint")
    business_standard = threading.Thread(target=business_standard_news, name="business_standard")
    economic_times = threading.Thread(target=economic_news, name="economic_times")
    money_control.start()
    mint.start()
    business_standard.start()
    economic_times.start()

    money_control.join()
    mint.join()
    business_standard.join()
    economic_times.join()
    print(len(mnews))
    print(len(ecnews))
    print(len(mcnews))
    print(len(bsnews))
    news = mcnews+mnews+ectimes+bsnews
    times = mtimes+mctimes+ectimes+bstimes
    data = pd.DataFrame({'Headline':news,'Date':times})
    data.to_csv('./raw_news/NewsData-' +
                      str(date.today()) + '.csv')
    
    # repo = git.Repo('.')
    # subprocess.check_output("git add .", stderr=subprocess.PIPE)
    # repo.index.commit('news fetch for ' + str(datetime.date.today()))
    # repo.remotes.origin.push()
    # -

