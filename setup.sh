#!/bin/bash

echo "🚀 SpectraShield Setup Script"
echo "=============================="

# Check prerequisites
echo "Checking prerequisites..."

command -v node >/dev/null 2>&1 || { echo "❌ Node.js is required but not installed. Aborting." >&2; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo "❌ Python 3 is required but not installed. Aborting." >&2; exit 1; }
command -v mongod >/dev/null 2>&1 || { echo "⚠️  MongoDB not found. Please install MongoDB." >&2; }
command -v redis-server >/dev/null 2>&1 || { echo "⚠️  Redis not found. Please install Redis." >&2; }

echo "✅ Prerequisites check complete"

# Setup environment
echo ""
echo "Setting up environment..."
if [ ! -f .env ]; then
    cp .env.example .env
    echo "✅ Created .env file from template"
else
    echo "ℹ️  .env file already exists"
fi

# Backend setup
echo ""
echo "Setting up backend..."
cd backend
npm install
echo "✅ Backend dependencies installed"
cd ..

# Frontend setup
echo ""
echo "Setting up frontend..."
cd frontend
npm install
echo "✅ Frontend dependencies installed"
cd ..

# ML Engine setup
echo ""
echo "Setting up ML Engine..."
cd ml-engine
pip3 install -r requirements.txt
echo "✅ ML Engine dependencies installed"
cd ..

# Create necessary directories
echo ""
echo "Creating directories..."
mkdir -p backend/uploads
mkdir -p backend/data
mkdir -p ml-engine/models
echo "✅ Directories created"

# Initialize blockchain
echo ""
echo "Initializing blockchain..."
cd backend
node scripts/init-blockchain.js
cd ..

echo ""
echo "=============================="
echo "✅ Setup complete!"
echo ""
echo "To start the application:"
echo "  1. Start MongoDB: mongod"
echo "  2. Start Redis: redis-server"
echo "  3. Start Backend: cd backend && npm run dev"
echo "  4. Start ML Engine: cd ml-engine && python api.py"
echo "  5. Start Frontend: cd frontend && npm run dev"
echo ""
echo "Or use Docker: docker-compose up"
echo ""
