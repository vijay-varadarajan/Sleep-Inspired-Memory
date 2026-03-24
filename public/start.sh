#!/bin/bash

# Sleep-Inspired Memory System - Startup Script

echo "🧠 Starting Sleep-Inspired Memory System..."
echo ""

# Check for GOOGLE_API_KEY
if [ -z "$GOOGLE_API_KEY" ]; then
    echo "⚠️  WARNING: GOOGLE_API_KEY environment variable not set!"
    echo "   The system will not function without it."
    echo ""
    echo "   Set it with: export GOOGLE_API_KEY='your-key-here'"
    echo ""
    read -p "   Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo "✅ API Key found"
fi

echo ""
echo "Starting Flask server..."
echo "Interface will be available at: http://localhost:5000"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Navigate to public directory and run
cd "$(dirname "$0")"
/home/spongyshaman/Documents/Sleep-Inspired-Memory/.venv/bin/python app.py
