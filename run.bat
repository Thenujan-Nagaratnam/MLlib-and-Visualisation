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

:: Set environment variables for PySpark
set PYSPARK_PYTHON=venv\Scripts\python.exe
set PYSPARK_DRIVER_PYTHON=venv\Scripts\python.exe


echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Installing required packages...
pip install -r requirements.txt

echo Starting Lyric Genre Classifier...
@REM streamlit run src/lyric_classifier.py
wave run src.app

pause