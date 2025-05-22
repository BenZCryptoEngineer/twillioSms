#!/bin/bash

# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    cp env.example .env
    echo "Created .env file from env.example. Please update it with your Twilio credentials."
fi

echo "Setup complete! You can now run the app with: python app.py" 