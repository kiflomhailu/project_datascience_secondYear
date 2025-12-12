#!/bin/bash

# Seismic Dashboard Launcher Script

echo "=================================="
echo "🚦 Seismic Risk Dashboard Launcher"
echo "=================================="
echo ""

# Check if we're in the right directory
if [ ! -d "dashboard" ]; then
    echo "❌ Error: dashboard folder not found"
    echo "   Please run this script from latest_cop directory"
    exit 1
fi

# Start API server in background
echo "📡 Starting Flask API server..."
cd dashboard/api

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python -m venv venv
    source venv/Scripts/activate
    echo "📦 Installing dependencies..."
    pip install -r requirements.txt
else
    source venv/Scripts/activate
fi

# Start API
python app.py &
API_PID=$!
echo "✅ API started (PID: $API_PID) - http://localhost:5000"

cd ../..

# Wait for API to be ready
echo "⏳ Waiting for API to be ready..."
sleep 5

# Start dashboard server in background
echo "🌐 Starting dashboard server..."
cd dashboard
python -m http.server 8080 &
DASH_PID=$!
echo "✅ Dashboard started (PID: $DASH_PID) - http://localhost:8080"

cd ..

echo ""
echo "=================================="
echo "✅ Dashboard is running!"
echo "=================================="
echo ""
echo "📊 Dashboard: http://localhost:8080"
echo "📡 API:       http://localhost:5000"
echo ""
echo "Press Ctrl+C to stop both servers"
echo "=================================="
echo ""

# Wait for user to press Ctrl+C
trap "echo ''; echo '🛑 Stopping servers...'; kill $API_PID $DASH_PID 2>/dev/null; echo '✅ Servers stopped'; exit 0" INT

# Keep script running
wait
