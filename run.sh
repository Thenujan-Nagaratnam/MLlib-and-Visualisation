#!/bin/bash

# Exit if any command fails
set -e

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
. venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Run the application
python3 app.py
