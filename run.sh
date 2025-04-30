#!/bin/bash

# Check if the virtual environment exists
if [ ! -d "venv" ]; then
    # Check if Python is installed
    if ! command -v python3 &> /dev/null; then
        echo "Python is not found. Please install Python and ensure it's added to the PATH."
        exit 1
    fi

    # Create the virtual environment
    python3 -m venv venv
fi

echo "Activating virtual environment..."
source venv/bin/activate

echo "Installing required packages..."
pip install -r requirements.txt

echo "Starting Lyric Genre Classifier..."
streamlit run src/lyric_classifier.py

# Keep the terminal open
read -p "Press any key to continue... "
