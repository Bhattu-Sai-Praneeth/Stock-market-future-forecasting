import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import datetime as dt
import os

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
    
    with st.spinner("Training model..."):
        model.fit(X_train, y_train, validation_data=(X_test, y_test), 
                  epochs=epochs, batch_size=32, verbose=1)
    
    train_pred = scaler.inverse_transform(model.predict(X_train))
    test_pred = scaler.inverse_transform(model.predict(X_test))
    
    y_train_actual = scaler.inverse_transform(y_train.reshape(-1, 1))
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    train_r2 = r2_score(y_train_actual, train_pred)
    test_r2 = r2_score(y_test_actual, test_pred)
    
    last_sequence = scaled_data[-time_step:]
    future_preds = predict_future(model, last_sequence, scaler)
    
    return train_r2, test_r2, future_preds

# Streamlit UI
st.title("Stock Market Prediction using LSTM")

# File upload section for the two required files
st.sidebar.header("Upload Files")
filtered_indices = st.sidebar.file_uploader("Upload filtered_indices_output.csv", type="csv")
sectors_file = st.sidebar.file_uploader("Upload sectors_with_symbols.csv", type="csv")

# Inform the user that daily data files are loaded automatically from the GitHub directory.
st.sidebar.info("Daily data files will be loaded automatically from the 'DATASETS/Daily_data/' directory.")

# Parameter for model training
epochs = st.sidebar.number_input("Number of Epochs", min_value=10, max_value=100, value=25)

if filtered_indices and sectors_file:
    try:
        # Load the uploaded files
        selected_indices = pd.read_csv(filtered_indices)
        sectors_df = pd.read_csv(sectors_file)
        
        # Load daily data files from the local directory
        daily_data = {}
        daily_data_folder = "DATASETS/Daily_data/"
        if os.path.exists(daily_data_folder):
            for file_name in os.listdir(daily_data_folder):
                if file_name.endswith('.csv'):
                    file_path = os.path.join(daily_data_folder, file_name)
                    # Convert filename to match the index name format (e.g., replacing underscores with dots)
                    name = os.path.splitext(file_name)[0].replace('_', '.')
                    daily_data[name] = pd.read_csv(file_path)
        else:
            st.error(f"Daily data folder '{daily_data_folder}' not found.")
            st.stop()
        
        results = []
        current_date = dt.datetime.now().strftime("%Y-%m-%d")
        
        # Process each index listed in the filtered_indices file
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
        
        # Display the prediction results if available
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
    st.info("Please upload the required files to begin processing")
