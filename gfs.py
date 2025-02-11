import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import datetime as dt
import requests
from io import StringIO

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

def fetch_csv_from_github(url):
    response = requests.get(url)
    if response.status_code == 200:
        try:
            # Use 'on_bad_lines' to skip problematic rows
            data = pd.read_csv(StringIO(response.text), on_bad_lines='skip')
            st.write(data.head())  # Log the first few rows to inspect the data
            return data
        except pd.errors.ParserError as e:
            st.error(f"Error parsing CSV: {str(e)}")
            return None
    else:
        st.error(f"Failed to fetch data from {url}")
        return None

# Streamlit UI
st.title("Stock Market Prediction using LSTM")

# File upload section
st.sidebar.header("Upload Files")
filtered_indices = st.sidebar.file_uploader("Upload filtered_indices_output.csv", type="csv")
sectors_file = st.sidebar.file_uploader("Upload sectors_with_symbols.csv", type="csv")

# Parameters
epochs = st.sidebar.number_input("Number of Epochs", min_value=10, max_value=100, value=25)

if filtered_indices and sectors_file:
    try:
        # Load data
        selected_indices = pd.read_csv(filtered_indices)
        sectors_df = pd.read_csv(sectors_file)
        
        # GitHub URL for daily data
        github_base_url = "https://github.com/Bhattu-Sai-Praneeth/test2/tree/main/DATASETS/Daily_data/"
        daily_files = ["AARTIIND_NS.csv", "ABBOTINDIA_NS.csv", "ADANIENT_NS.csv", "ADANIPORTS_NS.csv", "ALKEM_NS.csv", "APLAPOLLO_NS.csv", "APOLLOHOSP_NS.csv", "APOLLOTYRE_NS.csv", "ASHOKLEY_NS.csv", "ASIANPAINT_NS.csv", "AUBANK_NS.csv", "AUROPHARMA_NS.csv", "AXISBANK_NS.csv", "BAJAJ-AUTO_NS.csv", "BAJFINANCE_NS.csv", "BALKRISIND_NS.csv", "BALRAMCHIN_NS.csv", "BANDHANBNK_NS.csv", "BANKBARODA_NS.csv", "BATAINDIA_NS.csv", "BHARATFORG_NS.csv", "BHARTIARTL_NS.csv", "BIOCON_NS.csv", "BLUESTARCO_NS.csv", "BOSCHLTD_NS.csv", "BPCL_NS.csv", "BRIGADE_NS.csv", "BRITANNIA_NS.csv", "CENTURYPLY_NS.csv", "CERA_NS.csv", "CIPLA_NS.csv", "COALINDIA_NS.csv", "COLPAL_NS.csv", "CROMPTON_NS.csv", "DABUR_NS.csv", "DISHTV_NS.csv", "DIVISLAB_NS.csv", "DIXON_NS.csv", "DLF_NS.csv", "DRREDDY_NS.csv", "EICHERMOT_NS.csv", "EXIDEIND_NS.csv", "FEDERALBNK_NS.csv", "GLAND_NS.csv", "GLENMARK_NS.csv", "GODREJCP_NS.csv", "GODREJPROP_NS.csv", "GRANULES_NS.csv", "GRASIM_NS.csv", "HATHWAY_NS.csv", "HAVELLS_NS.csv", "HCLTECH_NS.csv", "HDFCBANK_NS.csv", "HDFCLIFE_NS.csv", "HEROMOTOCO_NS.csv", "HINDALCO_NS.csv", "HINDCOPPER_NS.csv", "HINDUNILVR_NS.csv", "HINDZINC_NS.csv", "ICICIBANK_NS.csv", "IDFCFIRSTB_NS.csv", "INDUSINDBK_NS.csv", "INFY_NS.csv", "IPCALAB_NS.csv", "ITC_NS.csv", "JBCHEPHARM_NS.csv", "JSL_NS.csv", "JSWSTEEL_NS.csv", "KAJARIACER_NS.csv", "KALYANKJIL_NS.csv", "KOTAKBANK_NS.csv", "LAURUSLABS_NS.csv", "LODHA_NS.csv", "LTIM_NS.csv", "LTTS_NS.csv", "LT_NS.csv", "LUPIN_NS.csv", "M&M_NS.csv", "MAHLIFE_NS.csv", "MANKIND_NS.csv", "MARICO_NS.csv", "MARUTI_NS.csv", "MOTHERSON_NS.csv", "MPHASIS_NS.csv", "MRF_NS.csv", "NATCOPHARM_NS.csv", "NAZARA_NS.csv", "NESTLEIND_NS.csv", "NETWORK18_NS.csv", "NMDC_NS.csv", "NTPC_NS.csv", "OBEROIRLTY_NS.csv", "ONGC_NS.csv", "PERSISTENT_NS.csv", "PGHH_NS.csv", "PHOENIXLTD_NS.csv", "PNB_NS.csv", "POLYCAB_NS.csv", "POWERGRID_NS.csv", "PRESTIGE_NS.csv", "PVRINOX_NS.csv", "RADICO_NS.csv", "RAJESHEXPO_NS.csv", "RATNAMANI_NS.csv", "RELIANCE_NS.csv", "SAIL_NS.csv", "SANOFI_NS.csv", "SAREGAMA_NS.csv", "SBILIFE_NS.csv", "SBIN_NS.csv", "SOBHA_NS.csv", "SUNPHARMA_NS.csv", "SUNTECK_NS.csv", "SUNTV_NS.csv", "TATACONSUM_NS.csv", "TATAMOTORS_NS.csv", "TATAMTRDVR_NS.csv", "TATASTEEL_NS.csv", "TCS_NS.csv", "TECHM_NS.csv", "TITAN_NS.csv", "TORNTPHARM_NS.csv", "TV18BRDCST_NS.csv", "TVSMOTOR_NS.csv", "UBL_NS.csv", "ULTRACEMCO_NS.csv", "UNITDSPR_NS.csv", "UPL_NS.csv", "VBL_NS.csv", "VEDL_NS.csv", "VGUARD_NS.csv", "VOLTAS_NS.csv", "WELCORP_NS.csv", "WHIRLPOOL_NS.csv", "WIPRO_NS.csv", "ZEEMEDIA_NS.csv", "ZYDUSLIFE_NS.csv", "^CNXAUTO.csv", "^CNXCONSUM.csv", "^CNXFMCG.csv", "^CNXIT.csv", "^CNXMEDIA.csv", "^CNXMETAL.csv", "^CNXPHARMA.csv", "^CNXREALTY.csv", "^NSEBANK.csv", "^NSEI.csv"]
  # Add all 147 file names
        
        daily_data = {}
        for file in daily_files:
            url = github_base_url + file
            df = fetch_csv_from_github(url)
            if df is not None:
                name = file.replace('.csv', '').replace('_', '.')
                daily_data[name] = df
        
        results = []
        current_date = dt.datetime.now().strftime("%Y-%m-%d")
        
        # Process each index
        for _, row in selected_indices.iterrows():
            index_name = row['indexname']
            if index_name in daily_data:
                company_name = sectors_df.loc[sectors_df['Index Name'] == index_name, 'Company Name'].iloc[0] if not sectors_df[sectors_df['Index Name'] == index_name].empty else index_name
                
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
    st.info("Please upload the required files to begin processing")
