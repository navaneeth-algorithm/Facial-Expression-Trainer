# 🎭 AI Face Expression Detection & Classification

A real-time facial emotion detection system that uses MediaPipe Face Mesh and TensorFlow.js KNN Classifier to detect and classify emotions from facial expressions. The system provides both web-based training and Python-based real-time detection capabilities.

## 📋 Table of Contents

- [Problem Statement](#-problem-statement)
- [Solution](#-solution)
- [Features](#-features)
- [Technology Stack](#-technology-stack)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [Usage](#-usage)
- [Screenshots](#-screenshots)
- [Contributing](#-contributing)
- [License](#-license)

## 🎯 Problem Statement

Traditional emotion detection systems often require:
- **Complex setup** - Difficult installation and configuration
- **Limited customization** - Pre-trained models with fixed emotions
- **Poor accuracy** - Generic models that don't adapt to individual users
- **No real-time capability** - Static image analysis only
- **Platform limitations** - Tied to specific frameworks or languages

## 💡 Solution

This project provides a **hybrid approach** combining:

### 🌐 **Web-Based Training Interface**
- **Interactive data collection** - Users train their own custom emotion models
- **Real-time feedback** - See emotions detected as you collect training data
- **Export capability** - Save trained models for use in other applications
- **Cross-platform** - Works in any modern web browser

### 🐍 **Python Real-Time Detection**
- **Live video analysis** - Real-time emotion detection from webcam
- **Custom model support** - Use models trained in the web interface
- **Visual feedback** - Emotion overlay with confidence scores
- **Cross-platform** - Works on Windows, macOS, and Linux

## ✨ Features

### 🎨 **Web Application Features**
- **Real-time face detection** using MediaPipe Face Mesh
- **Custom emotion training** with 30+ predefined emotions
- **Interactive data collection** with visual feedback
- **KNN Classifier** for personalized emotion recognition
- **Model export** to JSON format for Python usage
- **Beautiful UI** with glassmorphism design
- **Responsive layout** for desktop and mobile

### 🐍 **Python Application Features**
- **Real-time emotion detection** from webcam feed
- **Custom model loading** from exported JSON files
- **Visual emotion overlay** with emoji and confidence scores
- **All emotion percentages** display for detailed analysis
- **Screenshot capture** with timestamp
- **MediaPipe integration** with OpenCV fallback
- **Cross-platform compatibility**

### 🎯 **Core Capabilities**
- **52-dimensional blend shapes** from MediaPipe Face Mesh
- **K-Nearest Neighbors classification** for emotion prediction
- **Real-time processing** at 30 FPS
- **Confidence scoring** for prediction reliability
- **Dynamic emoji mapping** for 30+ emotions
- **Export/import functionality** for model portability

## 🛠 Technology Stack

### **Frontend (Web Application)**
- **HTML5** - Structure and layout
- **CSS3** - Styling with glassmorphism effects
- **JavaScript (ES6+)** - Application logic
- **MediaPipe Tasks Vision** - Face landmark detection
- **TensorFlow.js** - KNN Classifier implementation
- **Material Design Components** - UI components

### **Backend (Python Application)**
- **Python 3.8+** - Core language
- **OpenCV** - Computer vision and video processing
- **MediaPipe** - Face mesh detection (Python 3.8-3.11)
- **scikit-learn** - KNN Classifier
- **NumPy** - Numerical computations

### **Development Tools**
- **Node.js** - Local development server
- **http-server** - CORS-enabled file serving
- **Git** - Version control
- **Virtual environments** - Dependency isolation

## 📁 Project Structure

```
VisionEmojiClassification/
├── index.html              # Main web application
├── script.js              # Web application logic
├── style.css              # Application styling
├── package.json           # Node.js dependencies
├── start-server.sh        # Server startup script
├── SERVER_SETUP.md        # Server setup instructions
├── README.md              # Project documentation
├── .gitignore             # Git ignore rules
└── pythonTest/            # Python application
    ├── main.py            # Real-time detection script
    ├── demo_realtime.py   # Demo without trained model
    ├── requirements.txt   # Python dependencies
    ├── requirements-mediapipe.txt  # Full dependencies
    ├── setup_env.sh       # Automated setup (macOS/Linux)
    ├── setup_env.bat      # Automated setup (Windows)
    ├── setup.py           # Manual setup verification
    ├── README.md          # Python usage documentation
    ├── QUICKSTART.md      # Quick setup guide
    └── .gitignore         # Python-specific ignore rules
```

## 🚀 Quick Start

### **Web Application**
```bash
# Clone the repository
git clone <repository-url>
cd VisionEmojiClassification

# Start local server
npm install
./start-server.sh

# Open browser to http://localhost:8080
```

### **Python Application**
```bash
# Navigate to Python directory
cd pythonTest

# Automated setup (macOS/Linux)
./setup_env.sh

# Or manual setup
python3 -m venv emotion_env
source emotion_env/bin/activate
pip install -r requirements.txt

# Run real-time detection
python main.py path/to/exported_model.json
```

## 📖 Usage

### **1. Train Your Model (Web Application)**
1. **Open the web application** in your browser
2. **Enable webcam** and allow camera permissions
3. **Select emotions** from the dropdown menu
4. **Collect training data** by making facial expressions
5. **Train the model** when you have enough samples
6. **Export the model** to JSON format

### **2. Real-Time Detection (Python)**
1. **Activate virtual environment**
2. **Run the detection script** with your exported model
3. **Make facial expressions** in front of the webcam
4. **View real-time predictions** with confidence scores
5. **Save screenshots** or analyze emotion percentages

### **3. Demo Mode (No Model Required)**
```bash
# Test the setup without a trained model
python demo_realtime.py
```

## 🎥 Demo Video

Watch the project in action! This video demonstrates the complete workflow from training custom emotions to real-time detection.

[![AI Face Expression Detection Demo](https://img.youtube.com/vi/oyKTXbLogf0/maxresdefault.jpg)](https://www.youtube.com/watch?v=oyKTXbLogf0)

**Click the image above to watch the full demo on YouTube**

### **What You'll See in the Demo:**
- **Web Application Training** - Interactive emotion collection interface
- **Real-time Detection** - Live emotion classification with confidence scores
- **Python Application** - Cross-platform real-time video analysis
- **Model Export/Import** - Seamless workflow between web and Python
- **Custom Emotions** - Training personalized emotion recognition models

## 📸 Screenshots

### **Web Application**
- **Training Interface** - Interactive emotion collection
- **Real-time Detection** - Live emotion display
- **Model Export** - Save trained models

### **Python Application**
- **Real-time Video Feed** - Webcam with emotion overlay
- **Confidence Display** - All emotion percentages
- **Face Detection** - Green mesh overlay

## 🔧 Advanced Usage

### **Custom Emotion Training**
- **Add new emotions** to the dropdown menu
- **Collect diverse samples** for better accuracy
- **Balance training data** across all emotions
- **Export multiple models** for different use cases

### **Python Integration**
- **Load models** in your own Python applications
- **Batch processing** of images or videos
- **API development** using Flask or FastAPI
- **Data analysis** with pandas and matplotlib

## 🐛 Troubleshooting

### **Common Issues**
- **CORS errors** - Use the local server, not file:// protocol
- **MediaPipe installation** - Requires Python 3.8-3.11
- **Webcam permissions** - Allow camera access in browser
- **Model loading** - Ensure JSON file path is correct

### **Performance Optimization**
- **Good lighting** - Face the light source for better detection
- **Stable position** - Avoid rapid head movements
- **Sufficient training** - Collect 50+ samples per emotion
- **Balanced data** - Equal samples across all emotions

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. **Fork the repository**
2. **Create a feature branch**
3. **Make your changes**
4. **Add tests if applicable**
5. **Submit a pull request**

### **Areas for Contribution**
- **Additional emotion mappings**
- **Improved UI/UX design**
- **Performance optimizations**
- **Documentation improvements**
- **Cross-platform compatibility**

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### **What the MIT License Means:**
- ✅ **Free to use** - Anyone can use this software for any purpose
- ✅ **Free to modify** - You can change the code as needed
- ✅ **Free to distribute** - Share the software with others
- ✅ **Commercial use** - Use in commercial projects
- ✅ **Attribution required** - Must include the original license and copyright notice

**Perfect for open-source projects that want maximum compatibility and adoption!**

## 🙏 Acknowledgments

- **MediaPipe** - Google's framework for face mesh detection
- **TensorFlow.js** - Machine learning in the browser
- **OpenCV** - Computer vision library
- **scikit-learn** - Machine learning toolkit

## 📞 Support

If you encounter any issues or have questions:

1. **Check the troubleshooting section**
2. **Review the documentation**
3. **Open an issue** on GitHub
4. **Contact the maintainers**

---

**Built with ❤️ for the AI/ML community**

*Making emotion detection accessible, customizable, and real-time for everyone.*