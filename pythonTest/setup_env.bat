@echo off
REM setup_env.bat - Create and setup Python virtual environment for Windows

echo 🐍 Setting up Python Virtual Environment for Emotion Classifier
echo ================================================================

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python is not installed or not in PATH. Please install Python first.
    pause
    exit /b 1
)

echo ✅ Python found: 
python --version

REM Create virtual environment
echo 🔧 Creating virtual environment...
python -m venv emotion_env

if %errorlevel% neq 0 (
    echo ❌ Failed to create virtual environment
    pause
    exit /b 1
)

echo ✅ Virtual environment 'emotion_env' created successfully!

REM Activate virtual environment
echo 🚀 Activating virtual environment...
call emotion_env\Scripts\activate.bat

if %errorlevel% neq 0 (
    echo ❌ Failed to activate virtual environment
    pause
    exit /b 1
)

echo ✅ Virtual environment activated!

REM Upgrade pip
echo ⬆️  Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo 📦 Installing required packages...
pip install -r requirements.txt

if %errorlevel% neq 0 (
    echo ❌ Some packages failed to install
    echo 💡 Try installing core packages manually:
    echo    pip install numpy scikit-learn
)

echo.
echo 🎉 Setup completed successfully!
echo.
echo 📚 Next steps:
echo 1. Activate the environment: emotion_env\Scripts\activate.bat
echo 2. Export a model from the web application
echo 3. Run: python main.py path\to\exported_model.json
echo 4. Deactivate when done: deactivate
echo.
echo 💡 To reactivate the environment later, run:
echo    emotion_env\Scripts\activate.bat

pause
