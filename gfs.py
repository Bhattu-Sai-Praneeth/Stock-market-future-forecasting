import streamlit as st
import pandas as pd
import numpy as NaN
import datetime as dt
from pathlib import Path
import pandas_ta as ta
from tabulate import tabulate

# --- Functions ---
def append_row(df, row):
    return pd.concat([df, pd.DataFrame([row], columns=row.index)]).reset_index(drop=True)

def getRSI14(csvfilename):
    if Path(csvfilename).is_file():
        try:
            df = pd.read_csv(csvfilename)
            if df.empty:
                return 0.00, 0.00
            else:
                df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
                df['rsi14'] = ta.rsi(df['Close'], length=14)
                if pd.isna(df['rsi14'].iloc[-1]) or np.isnan(df['rsi14'].iloc[-1]):  # Use np.isnan
                    return 0.00, 0.00
                else:
                    rsival = df['rsi14'].iloc[-1].round(2)
                    ltp = df['Close'].iloc[-1].round(2)
                    return rsival, ltp
        except Exception as e:
            print(f"Error reading {csvfilename}: {e}")
            return 0.00, 0.00
    else:
        print(f"File does not exist: {csvfilename}")
        return 0.00, 0.00

def dayweekmonth_datasets(symbol, symbolname):
    if symbol.endswith('.NS'):
        symbol = symbol.replace(".NS", "_NS")

    base_path = Path("DATASETS")
    daylocationstr = base_path / "Daily_data" / f"{symbol}.csv"
    weeklocationstr = base_path / "Weekly_data" / f"{symbol}.csv"
    monthlocationstr = base_path / "Monthly_data" / f"{symbol}.csv"

    cday = dt.datetime.today().strftime('%d/%m/%Y')
    dayrsi14, dltp = getRSI14(daylocationstr)
    weekrsi14, wltp = getRSI14(weeklocationstr)
    monthrsi14, mltp = getRSI14(monthlocationstr)

    new_row = pd.Series({
        'entrydate': cday,
        'indexcode': symbol,
        'indexname': symbolname.strip(),
        'dayrsi14': dayrsi14,
        'weekrsi14': weekrsi14,
        'monthrsi14': monthrsi14
    })
    return new_row

def generateGFS(scripttype):
    indicesdf = pd.DataFrame(columns=['entrydate', 'indexcode', 'indexname', 'dayrsi14', 'weekrsi14', 'monthrsi14'])

    base_path = Path("DATASETS")
    fname = base_path / scripttype
    csvfilename = base_path / f"GFS_{scripttype}.csv"

    try:
        with open(fname) as f:
            for line in f:
                if "," not in line:
                    continue
                symbol, symbolname = line.split(",")[0], line.split(",")[1]
                symbol = symbol.strip()
                new_row = dayweekmonth_datasets(symbol, symbolname)
                indicesdf = append_row(indicesdf, new_row)
    except Exception as e:
        st.error(f"Error processing {fname}: {e}")
        return None

    return indicesdf


# --- Streamlit App ---
st.title("GFS Report Generator")

script_type = "indicesdf.csv"

if st.button("Generate GFS Report"):
    with st.spinner("Generating report..."):
        df = generateGFS(script_type)

        if df is not None:
            st.dataframe(df)
            st.write(tabulate(df, headers='keys', tablefmt='fancy_grid'))  # Display as a formatted table

            st.download_button(
                label="Download CSV",
                data=df.to_csv(index=False).encode('utf-8'),
                file_name=f"GFS_{script_type}.csv",
                mime="text/csv",
            )
        else:
            st.write("Report generation failed. Check error messages.")
