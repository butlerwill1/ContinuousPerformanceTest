#!/bin/bash
# Setup script for AX-CPT task

echo "Setting up AX-CPT virtual environment..."
echo ""

# Check if virtual environment already exists
if [ -d "venv" ]; then
    echo "Virtual environment already exists."
    echo "To recreate it, delete the 'venv' folder first."
    echo ""
else
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
    echo ""
fi

echo "To activate the virtual environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "Then install dependencies with:"
echo "  pip install -r requirements.txt"
echo ""
echo "Finally, run the task with:"
echo "  python main.py"

