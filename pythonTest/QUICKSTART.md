# 🚀 Quick Start Guide

## For macOS/Linux Users:

```bash
# 1. Navigate to pythonTest folder
cd pythonTest

# 2. Run automated setup
./setup_env.sh

# 3. Activate environment (if not already active)
source emotion_env/bin/activate

# 4. Export model from web app, then test it
python main.py ../emotion_classifier_2024-01-15.json

# 5. When done, deactivate
deactivate
```

## For Windows Users:

```cmd
REM 1. Navigate to pythonTest folder
cd pythonTest

REM 2. Run automated setup
setup_env.bat

REM 3. Activate environment (if not already active)
emotion_env\Scripts\activate.bat

REM 4. Export model from web app, then test it
python main.py ..\emotion_classifier_2024-01-15.json

REM 5. When done, deactivate
deactivate
```

## Manual Setup (All Platforms):

```bash
# 1. Create virtual environment
python3 -m venv emotion_env

# 2. Activate virtual environment
# macOS/Linux:
source emotion_env/bin/activate
# Windows:
emotion_env\Scripts\activate.bat

# 3. Install dependencies
pip install -r requirements.txt

# 4. Test the setup
python setup.py
```

## 🎯 What Each Script Does:

- **`setup_env.sh`** (macOS/Linux) - Creates venv, installs packages, shows next steps
- **`setup_env.bat`** (Windows) - Same as above but for Windows
- **`setup.py`** - Tests package imports and shows setup status
- **`main.py`** - Loads and uses exported emotion classifier models

## 🔧 Troubleshooting:

### Permission Issues (macOS/Linux):
```bash
chmod +x setup_env.sh
./setup_env.sh
```

### PowerShell Issues (Windows):
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Python Not Found:
- Install Python 3.7+ from [python.org](https://python.org)
- Make sure Python is in your system PATH

## 📚 Next Steps:

1. **Train a model** in the web application
2. **Export the model** using the Export button
3. **Test the model** with the Python scripts
4. **Integrate** with your own Python applications

Happy coding! 🐍✨
