#!/bin/bash

# Create a virtual environment named .venv
python -m venv .venv

# Activate the virtual environment
source .venv/bin/activate

# Install packages from requirements.txt
pip install -r requirements.txt

# Deactivate the environment (optional, for cleanup)
deactivate

echo "Virtual environment created and dependencies installed."   