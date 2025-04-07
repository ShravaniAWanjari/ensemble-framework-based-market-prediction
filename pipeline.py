import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import subprocess  # To run your base model scripts

# =============================================
# STEP 1: Streamlit UI for Inputs
# =============================================
st.title("Stock Trading Ensemble Predictor")

# Input widgets
ticker = st.text_input("Enter Stock Ticker (e.g., MSFT):", "MSFT")
prediction_date = st.date_input("Prediction Date:")
start_date = st.date_input("Start Date:")

if st.button("Run Prediction"):
    # =============================================
    # STEP 2: Fetch Data and Save to CSV
    # =============================================
    data = yf.download(ticker,start=start_date)
    data.reset_index(inplace=True)
    data.to_csv(f"{ticker}_data.csv", index=False)

    st.subheader(f"{ticker} Price Chart")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(data['Date'], data['Close'], color='royalblue')
    ax.set_xlabel("Date")
    ax.set_ylabel("Price ($)")
    ax.grid(True, linestyle='--')
    st.pyplot(fig)
    
    # =============================================
    # STEP 3: Run Base Models (LSTM, K-means, DQN)
    # =============================================
    # Replace with your actual scripts
    #with st.spinner("Running LSTM..."):
        #subprocess.run(["python", "base_models/lstm.py", ticker, str(prediction_date)])
    
    #with st.spinner("Detecting Market Regimes..."):
        #subprocess.run(["python", "base_models/kmeans.py", ticker])
    
    #with st.spinner("RL Agent Deciding..."):
        #subprocess.run(["python", "base_models/dqn.py", ticker])
    
    # =============================================
    # STEP 4: Load Predictions and Run Meta-Model
    # =============================================
    #lstm_preds = pd.read_csv(f"{ticker}_lstm_predictions.csv")
    #regimes = pd.read_csv(f"{ticker}_regime_labels.csv")
    #dqn_actions = pd.read_csv(f"{ticker}_dqn_actions.csv")
    
    # Merge data for the prediction date
    #merged = lstm_preds.merge(regimes, on="date").merge(dqn_actions, on="date")
    #merged = merged[merged["date"] == prediction_date.strftime("%Y-%m-%d")]
    
    # Load trained meta-model (pre-trained or retrain on the fly)
    # Example: Use XGBoost saved model
    #import joblib
    #meta_model = joblib.load("meta_model.pkl")
    
    # Generate final signal
    #features = merged[["lstm_pred", "regime_label", "dqn_action"]]
    #signal_prob = meta_model.predict_proba(features)[0][1]
    
    # Display results
    #st.subheader(f"Prediction for {ticker} on {prediction_date}:")
    #col1, col2, col3 = st.columns(3)
    #with col1:
        #st.metric("LSTM Predicted Price", f"${merged['lstm_pred'].values[0]:.2f}")
    #with col2:
        #st.metric("Market Regime", merged['regime_label'].map({0: "Sideways", 1: "Bull", 2: "Bear"}).values[0])
    #with col3:
        #st.metric("DQN Action", merged['dqn_action'].map({0: "Hold", 1: "Buy", 2: "Sell"}).values[0])
    
    # Final ensemble signal
    #if signal_prob > 0.7:
        #st.success(f"ENSEMBLE SIGNAL: BUY (Confidence: {signal_prob*100:.1f}%)")
    #elif signal_prob < 0.3:
        #st.error(f"ENSEMBLE SIGNAL: SELL (Confidence: {(1-signal_prob)*100:.1f}%)")
    #else:
        #st.warning("ENSEMBLE SIGNAL: HOLD")