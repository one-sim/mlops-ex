#!/bin/bash

# Create a virtual environment named .venv
python -m venv .venv

# Activate the virtual environment
source .venv/bin/activate

# Install packages from requirements.txt
pip install --upgrade pip
pip install -r requirements.txt

# Remove cached packages
pip cache purge

# Or clean system temp files
sudo rm -rf /tmp/*   

# Deactivate the environment (optional, for cleanup)
deactivate

echo "Virtual environment created and dependencies installed."   