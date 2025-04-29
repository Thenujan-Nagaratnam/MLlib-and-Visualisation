@echo off
IF NOT EXIST "venv" (

    :: Check if Python is installed
    where python >nul 2>nul
    IF ERRORLEVEL 1 (
        echo Python is not found. Please install Python and ensure it's added to the PATH.
        exit /b 1
    )

    :: Create the virtual environment
    python -m venv venv
)

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Installing required packages...
pip install -r requirements.txt

echo Starting Lyric Genre Classifier...
streamlit run src/lyric_classifier.py

pause