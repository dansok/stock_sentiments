# Stock Price Prediction Using LSTM

This repository contains a Python implementation for predicting stock prices using Long Short-Term Memory (LSTM) neural networks. The model leverages stock price data and news sentiment data to make predictions. The scope of this project is to attempt to predict AAPL prices over a period of 1 year. The following sections provide an overview of the project, including data preprocessing, model training, and result visualization.

## Scope

The scope of this project is to attempt to predict the price and volatility of AAPL stock over the calendar year 2023. The model uses historical stock price data and sentiment analysis from news articles to make predictions on a minute-by-minute basis. The goal is to provide accurate forecasts of closing prices and volatility to aid in financial analysis and decision-making.

## Project Structure

- **data/**: Directory containing stock price and sentiment data.
- **models/**: Directory where the trained model is saved.
- **main.py**: Main script to run the entire pipeline.
- **README.md**: Project documentation.

## Data

### Stock Data

The stock data is loaded from a directory of JSON files containing time series data for AAPL stock prices. Each JSON file represents 1-minute intervals of stock prices including open, high, low, close, and volume.

### Sentiment Data

The sentiment data is loaded from a JSON file containing sentiment scores from news articles. The sentiment scores are filtered to include only those related to AAPL and resampled to 1-minute intervals to align with the stock data.

## Data Preprocessing

1. **Load Stock Data**: The stock data is read from the JSON files and converted into a Pandas DataFrame.
2. **Load Sentiment Data**: The sentiment data is filtered to extract AAPL-related sentiment scores and resampled to match the stock data frequency.
3. **Merge Data**: The stock and sentiment data are merged into a single DataFrame. Additional features such as returns and volatility are calculated.
4. **Scaling**: The features are scaled using MinMaxScaler for normalization.

## Model

### LSTM Model

The LSTM model is implemented using PyTorch. The model consists of:
- An LSTM layer with specified input size, hidden size, and number of layers.
- A fully connected (linear) layer to map the LSTM output to the desired output size.

### Training

The model is trained using Mean Squared Error (MSE) loss and the Adam optimizer. The training loop runs for a specified number of epochs, updating the model weights based on the computed loss.

## Usage

### Prerequisites

- Python 3.7 or higher
- Required Python packages: numpy, pandas, torch, matplotlib, scikit-learn

### Running the Script

1. Place the stock data JSON files in the `data/aapl_prices` directory.
2. Place the sentiment data JSON file in the `data` directory.
3. Run the `main.py` script to train the model and visualize the results.

```bash
python main.py
```

### Output

- The trained LSTM model is saved in the `models` directory.
- Visualizations of the actual vs. predicted stock prices and volatilities are displayed.
- Mean Absolute Error (MAE) and Root Mean Square Error (RMSE) statistics are printed.

## Visualization

The script generates plots comparing the actual and predicted stock closing prices and volatilities. It also prints summary statistics for the prediction errors.

## Conclusion

This project demonstrates the application of LSTM networks to predict AAPL stock prices and volatility over the calendar year 2023. The model configuration used in this implementation required Kaggle TPUs for efficient training. 

Upon perusing the result figures, it becomes evident that the LSTM has effectively learned the trend components of both the price, and volatility time series. However, the volatility time series shows less accuracy, indicating that the model struggles to capture the stochastic noise component inherent in volatility data. This limitation aligns with intuition, as LSTM models are better suited for learning trends rather than capturing noise. 

Future work could focus on enhancing the model's ability to predict volatility by incorporating additional techniques tailored for modeling stochastic components in financial time series data.

