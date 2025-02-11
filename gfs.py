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
        predicted_value = next_pred[0, 0]
        predictions.append(predicted_value)
        current_sequence = np.roll(current_sequence, -1)
        current_sequence[-1] = predicted_value
    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

def get_predicted_values(data, epochs=25, start_date=None, end_date=None):
    df = data.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    
    # Filter data by selected date range if provided
    if start_date and end_date:
        df = df[(df['Date'] >= pd.to_datetime(start_date)) & (df['Date'] <= pd.to_datetime(end_date))]
    
    # Ensure there is enough data after filtering
    if len(df) < 50:
        st.warning("Not enough data in selected date range. Please select a wider range.")
        return None
    
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
    
    # Train model with a spinner indicator
    with st.spinner("Training model..."):
        model.fit(X_train, y_train, validation_data=(X_test, y_test), 
                  epochs=epochs, batch_size=32, verbose=1)
    
    train_pred = scaler.inverse_transform(model.predict(X_train))
    test_pred = scaler.inverse_transform(model.predict(X_test))
    
    y_train_actual = scaler.inverse_transform(y_train.reshape(-1, 1))
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    train_r2 = r2_score(y_train_actual, train_pred)
    test_r2 = r2_score(y_test_actual, test_pred)
    
    # Get dates for plotting
    train_dates = df['Date'].iloc[time_step + 1 : train_size]
    test_start = train_size + time_step
    test_end = test_start + len(y_test_actual)
    test_dates = df['Date'].iloc[test_start : test_end]
    
    last_sequence = scaled_data[-time_step:]
    future_preds = predict_future(model, last_sequence, scaler)
    
    return (
        train_r2, test_r2, future_preds,
        train_dates, y_train_actual, train_pred,
        test_dates, y_test_actual, test_pred
    )

# ------------------ Streamlit UI ------------------
st.title("Stock Market Prediction using LSTM")

# Sidebar: File Uploads
st.sidebar.header("Upload Files")
filtered_indices = st.sidebar.file_uploader("Upload filtered_indices_output.csv", type="csv")
sectors_file = st.sidebar.file_uploader("Upload sectors_with_symbols.csv", type="csv")
daily_files = st.sidebar.file_uploader("Upload Daily Data CSVs", type="csv", accept_multiple_files=True)

# Sidebar: Model Parameters
epochs = st.sidebar.number_input("Number of Epochs", min_value=10, max_value=150, value=25)

# Sidebar: Date Range Selector
st.sidebar.header("Select Date Range")
default_start = dt.date(2020, 1, 1)
default_end = dt.date.today()
start_date = st.sidebar.date_input("Start Date", value=default_start)
end_date = st.sidebar.date_input("End Date", value=default_end)

if start_date > end_date:
    st.sidebar.error("Error: End date must fall after start date.")

# Proceed only if all files are uploaded
if filtered_indices and sectors_file and daily_files:
    try:
        # Load data from uploaded files
        selected_indices = pd.read_csv(filtered_indices)
        sectors_df = pd.read_csv(sectors_file)
        
        daily_data = {}
        for file in daily_files:
            name = os.path.splitext(file.name)[0].replace('_', '.')
            daily_data[name] = pd.read_csv(file)
        
        results = []
        current_date = dt.datetime.now().strftime("%Y-%m-%d")
        
        # Process each index based on the selected indices file
        for _, row in selected_indices.iterrows():
            index_name = row['indexname']
            if index_name in daily_data:
                # Get company name from sectors file if available
                if not sectors_df[sectors_df['Index Name'] == index_name].empty:
                    company_name = sectors_df.loc[sectors_df['Index Name'] == index_name, 'Company Name'].iloc[0]
                else:
                    company_name = index_name
                
                st.subheader(f"Processing {index_name} ({company_name})")
                result = get_predicted_values(daily_data[index_name], epochs, start_date, end_date)
                if result is None:
                    continue  # Skip if not enough data
                
                (train_r2, test_r2, future_preds,
                 train_dates, y_train_actual, train_pred,
                 test_dates, y_test_actual, test_pred) = result
                
                # Create DataFrames for plotting (ensure Date is datetime)
                train_plot_df = pd.DataFrame({
                    'Date': pd.to_datetime(train_dates),
                    'Actual': y_train_actual.flatten(),
                    'Predicted': train_pred.flatten()
                })
                test_plot_df = pd.DataFrame({
                    'Date': pd.to_datetime(test_dates),
                    'Actual': y_test_actual.flatten(),
                    'Predicted': test_pred.flatten()
                })
                
                # Display training and test charts side-by-side
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"#### Training Data (R² = {train_r2:.4f})")
                    st.line_chart(train_plot_df.set_index('Date'))
                with col2:
                    st.write(f"#### Test Data (R² = {test_r2:.4f})")
                    st.line_chart(test_plot_df.set_index('Date'))
                
                # Generate future dates based on the last date in the dataset
                last_date_in_data = pd.to_datetime(daily_data[index_name]['Date']).max()
                future_dates = pd.date_range(last_date_in_data + pd.Timedelta(days=1), periods=5, freq='B')[:5]
                
                # Display future predictions chart
                future_plot_df = pd.DataFrame({
                    'Date': future_dates,
                    'Predicted': future_preds
                })
                st.write("#### Future Predictions")
                st.line_chart(future_plot_df.set_index('Date'))
                
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
        
        # Display the summary results table
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
            st.warning("No valid data found for processing.")
            
    except Exception as e:
        st.error(f"Error processing files: {str(e)}")
else:
    st.info("Please upload all required files to begin processing")
