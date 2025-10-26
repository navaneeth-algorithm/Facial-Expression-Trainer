#!/bin/bash
# setup_env.sh - Create and setup Python virtual environment

echo "🐍 Setting up Python Virtual Environment for Emotion Classifier"
echo "================================================================"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3 first."
    exit 1
fi

echo "✅ Python 3 found: $(python3 --version)"

# Create virtual environment
echo "🔧 Creating virtual environment..."
python3 -m venv emotion_env

if [ $? -eq 0 ]; then
    echo "✅ Virtual environment 'emotion_env' created successfully!"
else
    echo "❌ Failed to create virtual environment"
    exit 1
fi

# Activate virtual environment
echo "🚀 Activating virtual environment..."
source emotion_env/bin/activate

if [ $? -eq 0 ]; then
    echo "✅ Virtual environment activated!"
    echo "📍 You're now in the virtual environment (notice the 'emotion_env' prefix)"
else
    echo "❌ Failed to activate virtual environment"
    exit 1
fi

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📦 Installing required packages..."
pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✅ All packages installed successfully!"
else
    echo "❌ Some packages failed to install"
    echo "💡 Try installing core packages manually:"
    echo "   pip install numpy scikit-learn"
fi

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "📚 Next steps:"
echo "1. Activate the environment: source emotion_env/bin/activate"
echo "2. Export a model from the web application"
echo "3. Run: python main.py path/to/exported_model.json"
echo "4. Deactivate when done: deactivate"
echo ""
echo "💡 To reactivate the environment later, run:"
echo "   source emotion_env/bin/activate"
