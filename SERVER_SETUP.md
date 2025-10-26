# Simple HTTP Server Setup

## Option 1: Python Server (Recommended)
If you have Python installed:

```bash
# Navigate to your project folder
cd /Users/navaneethb/Desktop/programming/Machine\ Learning/VisionEmojiClassification

# Python 3
python3 -m http.server 8000

# Python 2
python -m SimpleHTTPServer 8000
```

Then open: http://localhost:8000

## Option 2: Node.js Server
If you have Node.js installed:

```bash
# Install a simple server
npm install -g http-server

# Navigate to your project folder
cd /Users/navaneethb/Desktop/programming/Machine\ Learning/VisionEmojiClassification

# Start server
http-server -p 8000
```

Then open: http://localhost:8000

## Option 3: VS Code Live Server
If you're using VS Code:
1. Install "Live Server" extension
2. Right-click on index.html
3. Select "Open with Live Server"

## Option 4: Simple PHP Server
If you have PHP installed:

```bash
cd /Users/navaneethb/Desktop/programming/Machine\ Learning/VisionEmojiClassification
php -S localhost:8000
```

Then open: http://localhost:8000

---

**The CORS error occurs because:**
- MediaPipe requires HTTPS or localhost
- Opening files directly (file://) doesn't work with MediaPipe
- You need to serve the files through a web server

**Try Option 1 (Python) first - it's the simplest!**
