#!/usr/bin/env python3
"""
Setup script for Python Emotion Classifier

This script helps set up the Python environment and test the installation.
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages from requirements.txt"""
    print("🔧 Installing Python dependencies...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        return False

def test_imports():
    """Test if all required packages can be imported"""
    print("🧪 Testing package imports...")
    
    required_packages = [
        ("numpy", "np"),
        ("sklearn", "sklearn"),
    ]
    
    optional_packages = [
        ("mediapipe", "mp"),
        ("cv2", "cv2"),
        ("matplotlib", "plt"),
    ]
    
    # Test required packages
    for package, alias in required_packages:
        try:
            exec(f"import {package} as {alias}")
            print(f"✅ {package} imported successfully")
        except ImportError:
            print(f"❌ {package} import failed")
            return False
    
    # Test optional packages
    for package, alias in optional_packages:
        try:
            exec(f"import {package} as {alias}")
            print(f"✅ {package} imported successfully (optional)")
        except ImportError:
            print(f"⚠️  {package} not available (optional)")
    
    return True

def main():
    """Main setup function"""
    print("🐍 Python Emotion Classifier Setup")
    print("=" * 40)
    
    # Check if requirements.txt exists
    if not os.path.exists("requirements.txt"):
        print("❌ requirements.txt not found!")
        print("Make sure you're running this script from the pythonTest directory.")
        sys.exit(1)
    
    # Install requirements
    if not install_requirements():
        sys.exit(1)
    
    # Test imports
    if not test_imports():
        sys.exit(1)
    
    print("\n🎉 Setup completed successfully!")
    print("\n📚 Next steps:")
    print("1. Export a model from the web application")
    print("2. Run: python main.py path/to/exported_model.json")
    print("3. Check README.md for detailed usage instructions")

if __name__ == "__main__":
    main()
