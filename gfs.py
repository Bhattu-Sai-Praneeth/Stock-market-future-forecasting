import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import datetime as dt
import os
import io

st.set_page_config(page_title="Stock Prediction", layout="wide")

def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

def predict_future(model, last_sequence, scaler, days=5):
    predictions = []
    current_sequence = last_sequence.copy()
    for _ in range(days):
        next_pred = model.predict(current_sequence.reshape(1, -1, 1))
        predictions.append(next_pred[0, 0])
        current_sequence = np.roll(current_sequence, -1)
        current_sequence[-1] = next_pred
    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

def get_predicted_values(data, epochs=25):
    df = data.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    
    close_data = df[['Close']]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_data)
    
    time_step = 13
    train_size = int(len(scaled_data) * 0.8)
    train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]
    
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, y_test = create_dataset(test_data, time_step)
    
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    model = Sequential([
        LSTM(32, return_sequences=True, input_shape=(time_step, 1)),
        LSTM(32, return_sequences=True),
        LSTM(32),
        Dense(1)
    ])
    model.compile(loss='mean_squared_error', optimizer='adam')
    
    with st.status("Training model...", expanded=True) as status:
        model.fit(X_train, y_train, validation_data=(X_test, y_test), 
                epochs=epochs, batch_size=32, verbose=1)
        status.update(label="Training complete", state="complete")
    
    train_pred = scaler.inverse_transform(model.predict(X_train))
    test_pred = scaler.inverse_transform(model.predict(X_test))
    
    y_train_actual = scaler.inverse_transform(y_train.reshape(-1, 1))
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    train_r2 = r2_score(y_train_actual, train_pred)
    test_r2 = r2_score(y_test_actual, test_pred)
    
    last_sequence = scaled_data[-time_step:]
    future_preds = predict_future(model, last_sequence, scaler)
    
    return train_r2, test_r2, future_preds

def is_valid_github_url(url):
    return url.startswith("https://raw.githubusercontent.com/")

# Streamlit UI
st.title("Stock Market Prediction using LSTM")

# File upload section
st.sidebar.header("Enter GitHub Raw URLs")
filtered_indices_url = st.sidebar.text_input("Filtered Indices CSV URL")
sectors_file_url = st.sidebar.text_input("Sectors CSV URL")
daily_files_urls = st.sidebar.text_area("Daily Data CSV URLs (one per line)")
epochs = st.sidebar.number_input("Number of Epochs", min_value=10, max_value=100, value=25)

if filtered_indices_url and sectors_file_url and daily_files_urls:
    try:
        # Validate URLs
        if not is_valid_github_url(filtered_indices_url) or \
           not is_valid_github_url(sectors_file_url):
            st.error("Invalid GitHub URL. Please use raw content URLs.")
            raise ValueError("Invalid GitHub URL")
        
        daily_files_list = [url.strip() for url in daily_files_urls.split('\n') if url.strip()]
        for url in daily_files_list:
            if not is_valid_github_url(url):
                st.error("Invalid GitHub URL in daily files. Please use raw content URLs.")
                raise ValueError("Invalid GitHub URL")

        # Load data from URLs
        with st.status("Loading data from GitHub...", expanded=True) as status:
            try:
                selected_indices = pd.read_csv(filtered_indices_url)
                sectors_df = pd.read_csv(sectors_file_url)
                status.update(label="Successfully loaded data.", state="complete")
            except Exception as e:
                st.error(f"Error loading data from GitHub: {str(e)}")
                raise

        # Process daily files from URLs
        daily_data = {}
        with st.status("Processing daily data files...", expanded=True) as status:
            for file_url in daily_files_list:
                try:
                    file_name = file_url.split('/')[-1].split('.')[0]
                    daily_data[file_name] = pd.read_csv(file_url)
                    print(f"Successfully read: {file_name}")
                except Exception as e:
                    st.error(f"Error reading daily data from {file_url}: {str(e)}")
                    raise
            status.update(label="Successfully processed daily data files.", state="complete")
        
        results = []
        current_date = dt.datetime.now().strftime("%Y-%m-%d")
        
        # Process each index
        for _, row in selected_indices.iterrows():
            index_name = row['indexname']
            if index_name in daily_data:
                company_name = sectors_df.loc[
                    sectors_df['Index Name'] == index_name, 'Company Name'
                ].iloc[0] if not sectors_df[sectors_df['Index Name'] == index_name].empty else index_name
                
                with st.expander(f"Processing {index_name} ({company_name})"):
                    train_r2, test_r2, future_preds = get_predicted_values(daily_data[index_name], epochs)
                    
                    results.append({
                        'Run Date': current_date,
                        'Index Name': index_name,
                        'Company Name': company_name,
                        'Model': 'LSTM',
                        'Train R2 Score': train_r2,
                        'Test R2 Score': test_r2,
                        'Day 1': future_preds[0],
                        'Day 2': future_preds[1],
                        'Day 3': future_preds[2],
                        'Day 4': future_preds[3],
                        'Day 5': future_preds[4]
                    })
        
        # Display results
        if results:
            result_df = pd.DataFrame(results)
            st.subheader("Prediction Results")
            st.dataframe(result_df.style.format({
                'Train R2 Score': '{:.4f}',
                'Test R2 Score': '{:.4f}',
                'Day 1': '{:.2f}',
                'Day 2': '{:.2f}',
                'Day 3': '{:.2f}',
                'Day 4': '{:.2f}',
                'Day 5': '{:.2f}'
            }))
        else:
            st.warning("No valid data found for processing")
            
    except Exception as e:
        st.error(f"Error processing files: {str(e)}")
else:
    st.info("Please enter all required GitHub raw content URLs to begin processing")
