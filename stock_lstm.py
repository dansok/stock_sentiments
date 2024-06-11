import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error


class StockLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(StockLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


def load_stock_data(stock_files_directory):
    all_data = []
    for filename in sorted(os.listdir(stock_files_directory)):
        filepath = os.path.join(stock_files_directory, filename)
        try:
            with open(filepath, "r") as file:
                stock_data_json = file.read()
                stock_data_dict = json.loads(stock_data_json)
                time_series = stock_data_dict["Time Series (1min)"]
                stock_df = pd.DataFrame.from_dict(time_series, orient="index")
                stock_df.index = pd.to_datetime(stock_df.index)
                stock_df = stock_df.rename(columns={
                    "1. open": "open",
                    "2. high": "high",
                    "3. low": "low",
                    "4. close": "close",
                    "5. volume": "volume"
                }).astype(float)
                all_data.append(stock_df)
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
    return pd.concat(all_data)


def load_sentiment_data(sentiment_file):
    sentiment_df = pd.read_json(sentiment_file)
    sentiment_df["time_published"] = pd.to_datetime(sentiment_df["time_published"], format="%Y%m%dT%H%M%S")

    # Filter sentiment for AAPL
    sentiment_df = sentiment_df[sentiment_df["ticker_sentiment"].apply(lambda x: any(d["ticker"] == "AAPL" for d in x))]

    # Extract relevant sentiment scores for AAPL
    def extract_aapl_sentiment(row):
        for ticker_info in row["ticker_sentiment"]:
            if ticker_info["ticker"] == "AAPL":
                return float(ticker_info["ticker_sentiment_score"])
        return None

    sentiment_df["aapl_sentiment_score"] = sentiment_df.apply(extract_aapl_sentiment, axis=1)
    sentiment_df = sentiment_df.set_index("time_published")[["aapl_sentiment_score"]].resample("1min").mean().fillna(0)
    return sentiment_df


def preprocess_data(stock_df, sentiment_df):
    merged_df = stock_df.join(sentiment_df["aapl_sentiment_score"], how="left")
    merged_df["aapl_sentiment_score"] = merged_df["aapl_sentiment_score"].fillna(0)  # Correctly filling NaNs with 0
    merged_df["returns"] = merged_df["close"].pct_change()
    merged_df["volatility"] = merged_df["returns"].rolling(window=21).std()
    return merged_df.dropna()


def create_sequences(data, seq_length):
    sequences = []
    targets = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i + seq_length])
        targets.append(data[i + seq_length, 0])
    return np.array(sequences), np.array(targets)


def train_lstm(X_train, y_train, input_size, hidden_size, num_layers, output_size, num_epochs, learning_rate):
    model = StockLSTM(input_size, hidden_size, num_layers, output_size).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

    return model


def visualize_results(model, X_train, y_train, scaler):
    model.eval()
    with torch.no_grad():
        predictions = model(X_train).cpu().numpy()

    # Inverse transform the predictions and actual values
    predictions = scaler.inverse_transform(
        np.concatenate((predictions, np.zeros((predictions.shape[0], 3))), axis=1)
    )[:, 0]
    y_train = scaler.inverse_transform(
        np.concatenate((y_train.cpu().numpy(), np.zeros((y_train.shape[0], 3))), axis=1)
    )[:, 0]

    # Plotting the results
    plt.figure(figsize=(14, 7))
    plt.plot(y_train, label="Actual")
    plt.plot(predictions, label="Predicted")
    plt.legend()
    plt.title("Actual vs Predicted Stock Prices")
    plt.show()

    # Summary error statistics
    mae = mean_absolute_error(y_train, predictions)
    rmse = np.sqrt(mean_squared_error(y_train, predictions))
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Root Mean Square Error (RMSE): {rmse:.4f}")


def visualize_volatility(model, seq_length, X_train, scaler, merged_df):
    model.eval()
    with torch.no_grad():
        predictions = model(X_train).cpu().numpy()

    # Inverse transform the predictions and actual values
    predictions = scaler.inverse_transform(
        np.concatenate((predictions, np.zeros((predictions.shape[0], 3))), axis=1)
    )[:, 3]  # Assuming the volatility is at the 4th column in the original scaled data

    # Extract realized volatility from merged DataFrame
    realized_volatility = merged_df["volatility"].iloc[seq_length:].values

    # Plotting the results
    plt.figure(figsize=(14, 7))
    plt.plot(realized_volatility, label="Realized Volatility", color='blue')  # Plot realized volatility in blue
    plt.plot(predictions, label="Predicted Volatility", color='orange')  # Plot predicted volatility in orange
    plt.legend()
    plt.title("Realized vs Predicted Volatility")
    plt.show()



def main():
    stock_files_directory = Path("data/aapl_prices")
    sentiment_file = Path("data/news_sentiment.json")

    # Load and preprocess data
    stock_df = load_stock_data(stock_files_directory=stock_files_directory)
    sentiment_df = load_sentiment_data(sentiment_file=sentiment_file)
    merged_df = preprocess_data(stock_df, sentiment_df)

    # Prepare data for LSTM
    features = merged_df[["close", "volume", "aapl_sentiment_score", "volatility"]]
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)

    seq_length = 30  # Example sequence length
    X, y = create_sequences(features_scaled, seq_length)

    # Convert data to PyTorch tensors
    X_train = torch.tensor(X, dtype=torch.float32).to(device)
    y_train = torch.tensor(y, dtype=torch.float32).to(device).view(-1, 1)

    # Train LSTM model
    input_size = X.shape[2]
    hidden_size = 128
    num_layers = 2
    output_size = 1
    num_epochs = 100
    learning_rate = 0.001

    model = train_lstm(X_train, y_train, input_size, hidden_size, num_layers, output_size, num_epochs, learning_rate)

    # Visualize results
    visualize_results(model, X_train, y_train, scaler)

    # Visualize realized vs predicted volatility
    visualize_volatility(model, seq_length, X_train, scaler, merged_df)

    # Save the model
    torch.save(model.state_dict(), Path("models/stock_lstm_model.pth"))


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main()
