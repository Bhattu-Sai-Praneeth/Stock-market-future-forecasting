import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime as dt
import os
from pathlib import Path
import pandas_ta as ta
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# -----------------------------
# Configuration & Paths
# -----------------------------
st.set_page_config(page_title="Stock Analysis & Prediction Dashboard", layout="wide")

# Set up directories (adjust BASE_DIR as needed)
BASE_DIR = r"C:\Users\saipr\OneDrive\Desktop\MAIN UPDATED PROJECT/DATASETS"
DAILY_DIR = os.path.join(BASE_DIR, "Daily_data")
WEEKLY_DIR = os.path.join(BASE_DIR, "Weekly_data")
MONTHLY_DIR = os.path.join(BASE_DIR, "Monthly_data")
SECTORS_FILE = os.path.join(BASE_DIR, "sectors with symbols.csv")

# -----------------------------
# --- GFS Analysis Functions ---
# -----------------------------
def get_latest_date():
    today = dt.date.today()
    return today.strftime("%Y-%m-%d")

def clean_and_save_data(data, filepath):
    data.reset_index(inplace=True)
    data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]
    for col in data.columns[1:]:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    data = data.dropna()
    data.to_csv(filepath, index=False)
    # st.write(f"Cleaned data saved to: {filepath}")

def download_stock_data(interval, folder):
    base_path = BASE_DIR
    filepath = os.path.join(base_path, "indicesstocks.csv")
    start_date = "2020-01-01"
    end_date = get_latest_date()

    # Count total symbols
    total_symbols = 0
    with open(filepath) as f:
        for line in f:
            if "," not in line:
                continue
            symbols = line.split(",")
            total_symbols += len([s.strip() for s in symbols if s.strip()])

    if total_symbols == 0:
        st.warning("No symbols found in indicesstocks.csv")
        return

    progress_bar = st.progress(0)
    status_text = st.empty()
    processed = 0

    with open(filepath) as f:
        for line in f:
            if "," not in line:
                continue
            symbols = line.split(",")
            for symbol in symbols:
                symbol = symbol.strip()
                if not symbol:
                    continue
                try:
                    data = yf.download(symbol, start=start_date, end=end_date, interval=interval)
                    ticketfilename = symbol.replace(".", "_")
                    save_path = os.path.join(base_path, folder, f"{ticketfilename}.csv")
                    clean_and_save_data(data, save_path)
                    processed += 1
                    progress = processed / total_symbols
                    progress_bar.progress(progress)
                    status_text.text(f"Downloading & Updating {folder} data: {processed}/{total_symbols} ({progress:.1%})")
                except Exception as e:
                    st.error(f"Error downloading & Updating {symbol}: {e}")
                    processed += 1
                    progress = processed / total_symbols
                    progress_bar.progress(progress)

    progress_bar.empty()
    status_text.empty()
    st.success(f"{folder.replace('_', ' ').title()} download & update completed!")

def fetch_vix():
    vix = yf.Ticker("^VIX")
    vix_data = vix.history(period="1d")
    return vix_data['Close'].iloc[0] if not vix_data.empty else None

def append_row(df, row):
    return pd.concat([df, pd.DataFrame([row], columns=row.index)]).reset_index(drop=True) if not row.isnull().all() else df

def getRSI14_and_BB(csvfilename):
    if Path(csvfilename).is_file():
        try:
            df = pd.read_csv(csvfilename)
            if df.empty or 'Close' not in df.columns:
                return 0.00, 0.00, 0.00, 0.00
            df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
            df['rsi14'] = ta.rsi(df['Close'], length=14)
            bb = ta.bbands(df['Close'], length=20)
            if bb is None or df['rsi14'] is None:
                return 0.00, 0.00, 0.00, 0.00
            df['lowerband'] = bb['BBL_20_2.0']
            df['middleband'] = bb['BBM_20_2.0']
            rsival = df['rsi14'].iloc[-1].round(2)
            ltp = df['Close'].iloc[-1].round(2)
            lowerband = df['lowerband'].iloc[-1].round(2)
            middleband = df['middleband'].iloc[-1].round(2)
            return rsival, ltp, lowerband, middleband
        except Exception as e:
            return 0.00, 0.00, 0.00, 0.00
    else:
        return 0.00, 0.00, 0.00, 0.00

def dayweekmonth_datasets(symbol, symbolname, index_code):
    symbol_with_underscore = symbol.replace('.', '_')
    day_path = os.path.join(DAILY_DIR, f"{symbol_with_underscore}.csv")
    week_path = os.path.join(WEEKLY_DIR, f"{symbol_with_underscore}.csv")
    month_path = os.path.join(MONTHLY_DIR, f"{symbol_with_underscore}.csv")
    cday = dt.datetime.today().strftime('%d/%m/%Y')
    dayrsi14, dltp, daylowerband, daymiddleband = getRSI14_and_BB(day_path)
    weekrsi14, wltp, weeklowerband, weekmiddleband = getRSI14_and_BB(week_path)
    monthrsi14, mltp, monthlowerband, monthmiddleband = getRSI14_and_BB(month_path)
    new_row = pd.Series({
        'entrydate': cday,
        'indexcode': index_code,
        'indexname': symbolname,
        'dayrsi14': dayrsi14,
        'weekrsi14': weekrsi14,
        'monthrsi14': monthrsi14,
        'dltp': dltp,
        'daylowerband': daylowerband,
        'daymiddleband': daymiddleband,
        'weeklowerband': weeklowerband,
        'weekmiddleband': weekmiddleband,
        'monthlowerband': monthlowerband,
        'monthmiddleband': monthmiddleband
    })
    return new_row

def generateGFS(scripttype):
    indicesdf = pd.DataFrame(columns=['entrydate', 'indexcode', 'indexname', 'dayrsi14', 
                                       'weekrsi14', 'monthrsi14', 'dltp', 'daylowerband', 
                                       'daymiddleband', 'weeklowerband', 'weekmiddleband', 
                                       'monthlowerband', 'monthmiddleband'])
    try:
        with open(os.path.join(BASE_DIR, f"{scripttype}.csv")) as f:
            for line in f:
                if "," not in line:
                    continue
                symbol, symbolname = line.split(",")[0], line.split(",")[1]
                new_row = dayweekmonth_datasets(symbol.strip(), symbolname.strip(), symbol.strip())
                indicesdf = append_row(indicesdf, new_row)
    except Exception as e:
        st.error(f"Error generating GFS report: {e}")
    return indicesdf

def read_indicesstocks(file_path):
    indices_dict = {}
    try:
        with open(file_path, 'r') as file:
            for line in file:
                parts = line.strip().split(',')
                if len(parts) > 1:
                    index_code = parts[0].strip()
                    indices_dict[index_code] = [stock.strip() for stock in parts[1:]]
    except Exception as e:
        st.error(f"Error reading indicesstocks.csv: {e}")
    return indices_dict

# -----------------------------
# --- LSTM Prediction Functions ---
# -----------------------------
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
    
    if start_date and end_date:
        df = df[(df['Date'] >= pd.to_datetime(start_date)) & (df['Date'] <= pd.to_datetime(end_date))]
    
    if len(df) < 50:
        st.warning("Not enough data in selected date range. Please select a wider range.")
        return None

    if 'High' in df.columns and 'Low' in df.columns:
        avg_high_gap = (df['High'] - df['Close']).mean()
        avg_low_gap = (df['Close'] - df['Low']).mean()
    else:
        avg_high_gap = 0
        avg_low_gap = 0
    
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
    
    train_dates = df['Date'].iloc[time_step + 1 : train_size]
    test_start = train_size + time_step
    test_end = test_start + len(y_test_actual)
    test_dates = df['Date'].iloc[test_start : test_end]
    
    last_sequence = scaled_data[-time_step:]
    future_preds = predict_future(model, last_sequence, scaler)
    
    future_high = future_preds + avg_high_gap
    future_low = future_preds - avg_low_gap
    
    return (train_r2, test_r2, future_preds, future_high, future_low,
            train_dates, y_train_actual, train_pred,
            test_dates, y_test_actual, test_pred)

# -----------------------------
# --- Streamlit App UI ---
# -----------------------------
st.title("Stock Analysis & Prediction Dashboard")
st.markdown("This app performs a two-step process: first it runs a GFS analysis to filter qualified indices/stocks, then it applies an LSTM model to predict future prices on the filtered stocks.")

# Sidebar (global settings, if needed)
st.sidebar.header("Navigation & Settings")
st.sidebar.markdown("""
- Use the **GFS Analysis** tab to download data, calculate technical indicators, and filter stocks.
- Use the **Stock Prediction (LSTM)** tab to configure and run future price predictions.
""")

# Use tabs so the user can run the two parts sequentially
tab1, tab2 = st.tabs(["GFS Analysis", "Stock Prediction (LSTM)"])

# -----------
# Tab 1: GFS Analysis
# -----------
with tab1:
    st.header("GFS Analysis")
    st.markdown("""
    **Overview:**  
    This section downloads stock data (Daily/Weekly/Monthly) for symbols listed in `indicesstocks.csv`, calculates technical indicators (RSI, Bollinger Bands), and filters stocks based on multi-timeframe criteria.
    """)
    if st.button("Run Full GFS Analysis"):
        with st.spinner("Fetching VIX data..."):
            vix_value = fetch_vix()
        if vix_value is None:
            st.error("Could not fetch VIX data. Please try again later.")
        elif vix_value > 20:
            st.warning(f"High Volatility (VIX: {vix_value:.2f}). Market not suitable for trading.")
        else:
            st.success(f"Market Volatility Normal (VIX: {vix_value:.2f}). Proceeding with analysis.")
            with st.spinner("Downloading and processing data..."):
                os.makedirs(DAILY_DIR, exist_ok=True)
                os.makedirs(WEEKLY_DIR, exist_ok=True)
                os.makedirs(MONTHLY_DIR, exist_ok=True)
                download_stock_data(interval='1d', folder='Daily_data')
                download_stock_data(interval='1wk', folder='Weekly_data')
                download_stock_data(interval='1mo', folder='Monthly_data')
                
                # Generate indices report from indicesdf.csv
                df3 = generateGFS('indicesdf')
                df4 = df3.loc[
                    df3['monthrsi14'].between(40, 60) &
                    df3['weekrsi14'].between(40, 60) &
                    df3['dayrsi14'].between(40, 60)
                ]
                st.markdown("### Qualified Indices")
                if df4.empty:
                    st.warning("No indices meet GFS criteria.")
                else:
                    st.dataframe(df4.style.format({
                        'dayrsi14': '{:.2f}',
                        'weekrsi14': '{:.2f}',
                        'monthrsi14': '{:.2f}',
                        'dltp': '{:.2f}'
                    }), use_container_width=True)
                    df4.to_csv(os.path.join(BASE_DIR, "filtered_indices.csv"), index=False)
                
                st.markdown("### Qualified Stocks")
                indices_dict = read_indicesstocks(os.path.join(BASE_DIR, "indicesstocks.csv"))
                results_df = pd.DataFrame(columns=df3.columns)
                for index in df4['indexcode']:
                    if index in indices_dict:
                        for stock in indices_dict[index]:
                            if stock:
                                new_row = dayweekmonth_datasets(stock, stock, index)
                                results_df = append_row(results_df, new_row)
                results_df = results_df.loc[
                    results_df['monthrsi14'].between(40, 60) &
                    results_df['weekrsi14'].between(40, 60) &
                    results_df['dayrsi14'].between(40, 60)
                ]
                if results_df.empty:
                    st.warning("No stocks meet GFS criteria.")
                else:
                    st.dataframe(results_df.style.format({
                        'dayrsi14': '{:.2f}',
                        'weekrsi14': '{:.2f}',
                        'monthrsi14': '{:.2f}',
                        'dltp': '{:.2f}'
                    }), use_container_width=True)
                    # Save the qualified stocks file for use in prediction
                    results_df.to_csv(os.path.join(BASE_DIR, "filtered_indices_output.csv"), index=False)
            st.success("GFS Analysis completed! Proceed to LSTM Analysis")

# -----------
# Tab 2: LSTM Stock Prediction
# -----------
with tab2:
    st.header("Stock Prediction using LSTM")
    st.markdown("""
    **Overview:**  
    This section loads the filtered stocks (output from the GFS analysis) and trains an LSTM model on their daily data to predict future prices.
    """)
    # Load the filtered indices file created from GFS analysis
    filtered_indices_path = os.path.join(BASE_DIR, "filtered_indices_output.csv")
    if os.path.exists(filtered_indices_path):
        selected_indices = pd.read_csv(filtered_indices_path)
        st.success("Loaded filtered indices from GFS Analysis.")
    else:
        st.error("Filtered indices file not found. Please run the GFS Analysis first.")
        st.stop()

    # Load sectors file from the local path
    if os.path.exists(SECTORS_FILE):
        sectors_df = pd.read_csv(SECTORS_FILE)
    else:
        st.error("sectors with symbols.csv file not found at the specified path.")
        st.stop()

    # Load daily data CSVs automatically from DAILY_DIR
    daily_data = {}
    daily_files_list = [f for f in os.listdir(DAILY_DIR) if f.endswith('.csv')]
    if not daily_files_list:
        st.error("No daily data files found in the Daily_data folder.")
        st.stop()
    for file in daily_files_list:
        file_path = os.path.join(DAILY_DIR, file)
        name = os.path.splitext(file)[0].replace('_', '.')
        try:
            df_daily = pd.read_csv(file_path)
            daily_data[name] = df_daily
        except Exception as e:
            st.error(f"Error loading {file}: {e}")

    # LSTM Configuration (displayed in the sidebar)
    st.sidebar.header("LSTM Configuration")
    epochs_input = st.sidebar.number_input("Number of Epochs", min_value=10, max_value=500, value=25)
    start_date = st.sidebar.date_input("Select Start Date", value=dt.date(2020, 1, 1))
    end_date = st.sidebar.date_input("Select End Date", value=dt.date.today())
    if start_date > end_date:
        st.error("Error: End date must fall after start date.")
        st.stop()

    # Run LSTM analysis button
    if st.button("Run LSTM Analysis"):
        results = []
        current_date = dt.datetime.now().strftime("%Y-%m-%d")
        for _, row in selected_indices.iterrows():
            index_name = row['indexname']
            if index_name in daily_data:
                # Get company name from sectors file if available
                if not sectors_df[sectors_df['Index Name'] == index_name].empty:
                    company_name = sectors_df.loc[sectors_df['Index Name'] == index_name, 'Company Name'].iloc[0]
                else:
                    company_name = index_name

                st.subheader(f"Processing {index_name} ({company_name})")
                result = get_predicted_values(daily_data[index_name], epochs=epochs_input, start_date=start_date, end_date=end_date)
                if result is None:
                    st.warning(f"Not enough data for {index_name}. Skipping...")
                    continue
                (train_r2, test_r2, future_preds, future_high, future_low,
                 train_dates, y_train_actual, train_pred,
                 test_dates, y_test_actual, test_pred) = result

                # Create DataFrames for plotting
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

                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"#### Training Data (R² = {train_r2:.4f})")
                    st.line_chart(train_plot_df.set_index('Date'))
                with col2:
                    st.write(f"#### Test Data (R² = {test_r2:.4f})")
                    st.line_chart(test_plot_df.set_index('Date'))

                last_date_in_data = pd.to_datetime(daily_data[index_name]['Date']).max()
                future_dates = pd.date_range(last_date_in_data + pd.Timedelta(days=1), periods=5, freq='B')[:5]
                future_plot_df = pd.DataFrame({
                    'Date': future_dates,
                    'Predicted Close': future_preds,
                    'Predicted High': future_high,
                    'Predicted Low': future_low
                })
                st.write("#### Future Predictions")
                st.line_chart(future_plot_df.set_index('Date'))

                # Append the results for this company
                results.append({
                    'Run Date': current_date,
                    'Index Name': index_name,
                    'Company Name': company_name,
                    'Model': 'LSTM',
                    'Train R2 Score': train_r2,
                    'Test R2 Score': test_r2,
                    'Close Day 1': future_preds[0],
                    'Close Day 2': future_preds[1],
                    'Close Day 3': future_preds[2],
                    'Close Day 4': future_preds[3],
                    'Close Day 5': future_preds[4],
                    'High Day 1': future_high[0],
                    'High Day 2': future_high[1],
                    'High Day 3': future_high[2],
                    'High Day 4': future_high[3],
                    'High Day 5': future_high[4],
                    'Low Day 1': future_low[0],
                    'Low Day 2': future_low[1],
                    'Low Day 3': future_low[2],
                    'Low Day 4': future_low[3],
                    'Low Day 5': future_low[4]
                })

        if results:
            result_df = pd.DataFrame(results)
            st.subheader("Prediction Results Summary")
            st.dataframe(result_df.style.format({
                'Train R2 Score': '{:.4f}',
                'Test R2 Score': '{:.4f}',
                'Close Day 1': '{:.2f}',
                'Close Day 2': '{:.2f}',
                'Close Day 3': '{:.2f}',
                'Close Day 4': '{:.2f}',
                'Close Day 5': '{:.2f}',
                'High Day 1': '{:.2f}',
                'High Day 2': '{:.2f}',
                'High Day 3': '{:.2f}',
                'High Day 4': '{:.2f}',
                'High Day 5': '{:.2f}',
                'Low Day 1': '{:.2f}',
                'Low Day 2': '{:.2f}',
                'Low Day 3': '{:.2f}',
                'Low Day 4': '{:.2f}',
                'Low Day 5': '{:.2f}'
            }))

            # Generate verdict for each company based on Test R2 Score
            result_df['Verdict'] = result_df['Test R2 Score'].apply(
                lambda x: "Strong Forecast" if x >= 0.8 else ("Moderate Forecast" if x >= 0.6 else "Weak Forecast")
            )

            # Create a verdict table with selected columns
            verdict_df = result_df[['Index Name', 'Company Name', 'Test R2 Score', 'Verdict']]
            st.subheader("Company Verdict")
            st.dataframe(verdict_df.style.format({'Test R2 Score': '{:.4f}'}))
        else:
            st.warning("No valid data found for prediction.")
