# Stock Market Future Forecasting

A Streamlit dashboard that combines technical analysis (RSI and Bollinger Bands across daily, weekly and monthly data), news sentiment analysis (VADER or FinBERT), and an LSTM model for future price predictions.

## Requirements

- Python 3.11 (the dependency versions are pinned for it)
- The bundled sample data in `DATASETS/`

## Setup

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m nltk.downloader vader_lexicon
```

On Windows, `package_downloader.bat` runs the same commands.

The FinBERT sentiment option downloads the `ProsusAI/finbert` model from Hugging Face the first time it is selected. VADER only needs the NLTK lexicon installed above.

## Run

```bash
python -m streamlit run gfs.py
```

`Main.py`, `Main2.py` and `Main3.py` are alternative dashboard versions. `summary.py` provides a Gradio interface:

```bash
python summary.py
```

## Data folder

The scripts read and write data in `DATASETS/` next to the source files. To use a different folder, set the `STOCK_DATA_DIR` environment variable to its path.
