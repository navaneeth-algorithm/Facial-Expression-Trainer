#!/bin/bash
# Start script for Face Expression Detection

echo "🚀 Starting Face Expression Detection Server..."
echo "📁 Project directory: $(pwd)"
echo ""

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed!"
    echo "Please install Node.js from: https://nodejs.org/"
    exit 1
fi

# Check if npm is installed
if ! command -v npm &> /dev/null; then
    echo "❌ npm is not installed!"
    echo "Please install npm (comes with Node.js)"
    exit 1
fi

echo "✅ Node.js version: $(node --version)"
echo "✅ npm version: $(npm --version)"
echo ""

# Install http-server if not already installed
if ! npm list -g http-server &> /dev/null; then
    echo "📦 Installing http-server..."
    npm install -g http-server
fi

echo "🌐 Starting server on http://localhost:8000"
echo "📱 Open your browser and go to: http://localhost:8000"
echo "🛑 Press Ctrl+C to stop the server"
echo ""

# Start the server
http-server -p 8000 -c-1 -o
