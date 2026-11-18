@echo off
setlocal
cd /d "%~dp0"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m nltk.downloader vader_lexicon
echo Dependencies installed. Start the app with: python -m streamlit run gfs.py
