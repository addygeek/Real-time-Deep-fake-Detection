#!/bin/bash

echo "========================================"
echo "SpectraShield - Starting All Services"
echo "========================================"

echo ""
echo "Checking if services are installed..."

# Check if backend node_modules exists
if [ ! -d "backend/node_modules" ]; then
    echo "Installing backend dependencies..."
    cd backend
    npm install
    cd ..
fi

# Check if frontend node_modules exists
if [ ! -d "frontend/node_modules" ]; then
    echo "Installing frontend dependencies..."
    cd frontend
    npm install
    cd ..
fi

echo ""
echo "========================================"
echo "Starting Services..."
echo "========================================"

echo ""
echo "[1/3] Starting Backend on port 4000..."
cd backend
npm run dev &
BACKEND_PID=$!
cd ..
sleep 3

echo "[2/3] Starting ML Engine on port 5000..."
cd ml-engine
python api.py &
ML_PID=$!
cd ..
sleep 3

echo "[3/3] Frontend already running on port 3000..."
echo ""

echo "========================================"
echo "All Services Started!"
echo "========================================"
echo ""
echo "Access the application:"
echo "  Frontend: http://localhost:3000"
echo "  Backend:  http://localhost:4000"
echo "  ML API:   http://localhost:5000"
echo ""
echo "Process IDs:"
echo "  Backend: $BACKEND_PID"
echo "  ML Engine: $ML_PID"
echo ""

echo "Checking service health..."
sleep 2
curl -s http://localhost:4000/health
echo ""
curl -s http://localhost:5000/health
echo ""

echo ""
echo "========================================"
echo "Services are running!"
echo "Press Ctrl+C to stop all services"
echo "========================================"

# Wait for user interrupt
trap "kill $BACKEND_PID $ML_PID; exit" INT
wait
