# Stock Market Prediction Using Sentiment Analysis

## Overview
This project aims to predict the stock price of a particular Indian stock (e.g., Reliance Industries) using sentiment analysis of news articles. By leveraging historical stock prices and sentiment extracted from news articles, we aim to improve predictive accuracy using an LSTM model.

---

## Features
- **Stock Price Prediction**: Predicts the closing price of the stock based on historical data and sentiment.
- **Sentiment Analysis**: Analyzes news articles to extract sentiment scores for a specific stock or its sector.
- **Time-Series Modeling**: Uses Long Short-Term Memory (LSTM) networks for predictive modeling.

---

## Data Sources
### News Articles
The project uses NewsCatcherAPI[https://www.newscatcherapi.com/](https://newscatcherapi.com/) to fetch news articles:
- Fetch articles based on keywords (e.g., "Reliance Industries").
- Filter articles by publication date to align with stock data.
- Sentiment analysis is performed on the extracted articles.

### Stock Prices
Historical stock prices are obtained from sources like Yahoo Finance or NSE/BSE websites.

---

## Workflow
1. **Data Collection**
    - Use the Newscatcher API to gather news articles related to the stock.
    - Collect historical stock prices for the target company.

2. **Sentiment Analysis**
    - Perform text preprocessing on the news articles (e.g., tokenization, stopword removal).
    - Use a sentiment analysis library like VADER, TextBlob, or a pretrained transformer model to score sentiment.

3. **Feature Engineering**
    - Align news sentiment with corresponding stock price dates.
    - Fill gaps for dates with no news by using neutral sentiment or sector news sentiment.

4. **Modeling**
    - Preprocess the time series data (e.g., normalization, train-test split).
    - Train an LSTM model using features:
        - Historical stock prices.
        - Sentiment scores.

5. **Evaluation**
    - Evaluate model performance using metrics like Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).

---

## Installation and Usage
### Prerequisites
- Python 3.8+
- API Key for Newscatcher API (Sign up [here](https://newscatcherapi.com/))

### Installation
```bash
# Clone the repository
git clone https://github.com/ABW1729/stock-market-prediction.git

# Navigate to the project directory
cd stock-prediction-sentiment

# Install dependencies
pip install -r requirements.txt
```

### Configuration
1. Obtain your API key from Newscatcher.
2. Update the `config.json` file with your API key and target keywords.

```json
{
  "api_key": "your_newscatcher_api_key",
  "keywords": ["Reliance Industries", "energy sector"]
}
```

### Run the Project
```bash
# Fetch news articles
python fetch_news.py

# Perform sentiment analysis
python sentiment_analysis.py

# Train and evaluate the LSTM model
python train_model.py
```

---

## Results
- The model predicts stock prices with an MAE of X%.
- Sentiment analysis significantly improved prediction accuracy on days with stock-specific news.

---

## References
- [Newscatcher API Documentation](https://newscatcherapi.com/documentation)
- [LSTM Networks](https://en.wikipedia.org/wiki/Long_short-term_memory)
- [TextBlob for Sentiment Analysis](https://textblob.readthedocs.io/en/dev/)

---

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for improvements.

---

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments
- The Newscatcher API team for providing access to comprehensive news data.
- OpenAI for providing GPT models to assist in project planning and implementation.
