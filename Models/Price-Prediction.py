# Imports

import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU
from tensorflow.keras.callbacks import EarlyStopping


# select the stock from yfinance

ticker = 'AAPL'
data = yf.download(ticker)
close_prices = data[['Close']]

# data splits and preprocessing

scaler = MinMaxScaler(feature_range=(0, 1))
prices_scaled = scaler.fit_transform(close_prices.values)

def create_dataset(dataset, time_step=60):
    X, y = [], []
    for i in range(len(dataset) - time_step):
        X.append(dataset[i:i + time_step, 0])
        y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(y)

time_step = 60
X, y = create_dataset(prices_scaled, time_step)
X = X.reshape(X.shape[0], X.shape[1], 1)

train_size = int(len(X) * 0.8)
x_train, x_val = X[:train_size], X[train_size:]
y_train, y_val = y[:train_size], y[train_size:]

# GRU model for the price prediction

gru_model = Sequential([
    GRU(50, return_sequences=True, input_shape=(time_step, 1)),
    GRU(50),
    Dense(25),
    Dense(1)
])
gru_model.compile(optimizer='adam', loss='mean_squared_error')
gru_model.summary()

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history = gru_model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=50,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

gru_x_val=x_val
gru_y_val=y_val
gru_y_pred_scaled = gru_model.predict(gru_x_val)
gru_y_pred = scaler.inverse_transform(gru_y_pred_scaled)
y_actual = scaler.inverse_transform(y_val.reshape(-1, 1))

# Error metrics for GRU

gru_rmse = np.sqrt(mean_squared_error(y_actual, gru_y_pred))
gru_mae = mean_absolute_error(y_actual, gru_y_pred)
gru_r2 = r2_score(y_actual, gru_y_pred)
print(f'Validation RMSE: {gru_rmse:.4f} USD')
print(f'Validation MAE: {gru_mae:.4f} USD')
print(f'Validation R²: {gru_r2:.4f}')


# LSTM for the same price prediction thiing

lstm_model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(time_step, 1)),
    LSTM(50),
    Dense(25),
    Dense(1)
])
lstm_model.compile(optimizer='adam', loss='mean_squared_error')
lstm_model.summary()

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history = lstm_model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=50,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# Show the LSTM graph

lstm_x_val=x_val
lstm_y_val=y_val
lstm_y_pred_scaled = lstm_model.predict(lstm_x_val)
lstm_y_pred = scaler.inverse_transform(lstm_y_pred_scaled)
y_actual = scaler.inverse_transform(y_val.reshape(-1, 1))


# Error metrics for LSTM

lstm_rmse = np.sqrt(mean_squared_error(y_actual, lstm_y_pred))
lstm_mae = mean_absolute_error(y_actual, lstm_y_pred)
lstm_r2 = r2_score(y_actual, lstm_y_pred)
print(f'Validation RMSE: {lstm_rmse:.4f} USD')
print(f'Validation MAE: {lstm_mae:.4f} USD')
print(f'Validation R²: {lstm_r2:.4f}')


y_lstm_pred = scaler.inverse_transform(lstm_model.predict(x_val))
y_gru_pred = scaler.inverse_transform(gru_model.predict(x_val))
y_actual = scaler.inverse_transform(y_val.reshape(-1, 1))

# Create a figure for the plot
plt.figure(figsize=(14, 6))
plt.plot(y_actual, label='Actual Price', linewidth=2)
plt.plot(y_lstm_pred, label='LSTM Prediction')
plt.plot(y_gru_pred, label='GRU Prediction')
plt.title('Stock Price Prediction: LSTM vs GRU')
plt.xlabel('Time')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True)

# Save the plot to a file instead of showing it
plt.savefig(f'Results/{ticker}_Price_Comparison.png')  # Save as PNG

# comparision of metrics
lstm_rmse = np.sqrt(mean_squared_error(y_actual, lstm_y_pred))
gru_rmse = np.sqrt(mean_squared_error(y_actual, gru_y_pred))

lstm_mae = mean_absolute_error(y_actual, lstm_y_pred)
gru_mae = mean_absolute_error(y_actual, gru_y_pred)

lstm_r2 = r2_score(y_actual, lstm_y_pred)
gru_r2 = r2_score(y_actual, gru_y_pred)
# Define a scoring function
def calculate_score(rmse, mae, r2):
    # You can adjust the weights as needed
    return (1 / rmse) + (1 / mae) + r2  # Higher is better

# Calculate scores for both models
lstm_score = calculate_score(lstm_rmse, lstm_mae, lstm_r2)
gru_score = calculate_score(gru_rmse, gru_mae, gru_r2)

# Determine the better model based on the score
if lstm_score > gru_score:
    better_model_results = {
        'RMSE': lstm_rmse,
        'MAE': lstm_mae,
        'R² Score': lstm_r2,
        'Predicted': lstm_y_pred.flatten()
    }
else:
    better_model_results = {
        'RMSE': gru_rmse,
        'MAE': gru_mae,
        'R² Score': gru_r2,
        'Predicted': gru_y_pred.flatten()
    }

# Create a DataFrame from the better model results
# Create a DataFrame from the better model results with dates as index
results_df = pd.DataFrame({
    'Actual_Price': y_actual.flatten(),
    'Predicted': better_model_results['Predicted']
}, index=data.index[-len(y_actual):])  # Use the last len(y_actual) dates from the original data

# Display first few rows
print(results_df.head())

# Save to CSV
results_df.to_csv(f'Results/Price_predictions.csv', index=True)  # Save with index
print(f"\n* Saved predictions to Results/Price_predictions.csv")  # Updated print statement