# Stock News Analyzer
[![run Stock.py](https://github.com/KingDEV22/Stock-Analysis/actions/workflows/actions.yml/badge.svg)](https://github.com/KingDEV22/Stock-Analysis/actions/workflows/actions.yml)


Stock News Analyzer is a Python-based project that aims to provide sentiment analysis of stock-related news articles. It utilizes web scraping techniques using Beautiful Soup to extract news articles from top news websites. The project incorporates various deep learning algorithms such as LSTM, BiGRU, and machine learning models like logistic regression to predict the sentiment of the news.

## Features

The Stock News Analyzer project includes the following key features:

- **Web Scraping**: The project uses Beautiful Soup, a Python library, to scrape stock-related news articles from top news websites. This ensures the availability of up-to-date and relevant news for sentiment analysis.

- **Sentiment Analysis**: The project employs a sentiment analysis model to predict the sentiment of the news articles. The sentiment can be categorized as positive (indicating potential profit in the company's stock), neutral (suggesting to hold the stock), or negative (indicating potential loss).

- **Multiple Models**: Various deep learning algorithms, including LSTM and BiGRU, are implemented to train sentiment analysis models. Additionally, a machine learning model like logistic regression is utilized for prediction. The project compares the predictions from all the models and determines the overall sentiment using a voting mechanism.

- **Automated News Retrieval**: The project utilizes GitHub Actions to fetch stock-related news on a daily basis. This ensures that the sentiment analysis is performed on the latest news articles, keeping the analysis up-to-date and relevant.

## Setup and Installation

To set up the Stock News Analyzer project locally, follow these steps:

1. Clone the repository: `git clone https://github.com/your-username/stock-news-analyzer.git`
2. Navigate to the project directory: `cd stock-news-analyzer`
3. Install the required dependencies: `pip install -r requirements.txt`
4. Execute the main script to fetch news and perform sentiment analysis: `python Stocks.py`

Please note that additional configuration might be required based on your specific environment and data sources. Refer to the project's documentation for detailed instructions.

## Future Scope

The Stock News Analyzer project is still under development, and future enhancements are planned. Some potential areas of improvement and expansion include:

- **Integration with Additional Data**: The project aims to integrate more data sources, such as historical stock data and financial indicators, to analyze and determine stock trends more accurately. This would provide users with a more comprehensive understanding of the market.

- **Visualization**: Incorporating data visualization techniques can enhance the presentation of sentiment analysis results. Graphs, charts, and interactive visualizations can help users better interpret and analyze the sentiment trends over time.

- **User Interface**: Building a user-friendly interface to access and interact with the sentiment analysis results can improve the overall usability and accessibility of the project. Users will be able to easily view and navigate through the analyzed news articles and associated sentiment predictions.



