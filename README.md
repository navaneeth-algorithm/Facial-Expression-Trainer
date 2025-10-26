# Face Mesh Expression Detection

A real-time facial expression detection application using MediaPipe Face Mesh and vanilla JavaScript.

## Features

- **Real-time Face Detection**: Uses MediaPipe Face Mesh to detect facial landmarks
- **Expression Classification**: Identifies 5 basic emotions:
  - 😊 Happy
  - 😢 Sad
  - 😠 Angry
  - 😲 Surprised
  - 😐 Neutral
- **Confidence Scoring**: Shows detection confidence percentage
- **Visual Feedback**: Displays face mesh overlay and landmark points
- **Responsive Design**: Works on desktop and mobile devices

## How It Works

The application analyzes facial landmarks to determine expressions:

1. **Eye Aspect Ratio (EAR)**: Detects eye openness for surprise/sadness
2. **Mouth Aspect Ratio (MAR)**: Detects smile intensity for happiness
3. **Eyebrow Height**: Detects raised eyebrows for surprise
4. **Facial Tension**: Analyzes overall facial muscle tension

## Setup

1. **Clone or download** the project files
2. **Open `index.html`** in a modern web browser
3. **Grant camera permissions** when prompted
4. **Click "Start Detection"** to begin

## Requirements

- Modern web browser with camera support
- HTTPS connection (required for camera access in production)
- WebGL support (for MediaPipe)

## Browser Compatibility

- Chrome 80+
- Firefox 75+
- Safari 13+
- Edge 80+

## Technical Details

- **MediaPipe Face Mesh**: Provides 468 facial landmarks
- **Real-time Processing**: ~30 FPS on modern devices
- **No External Dependencies**: Pure JavaScript implementation
- **Responsive Canvas**: Automatically adjusts to video dimensions

## Customization

You can modify the expression detection logic in `script.js`:

- Adjust confidence thresholds
- Add new expression types
- Modify landmark analysis algorithms
- Change visual styling

## Troubleshooting

- **Camera not working**: Ensure HTTPS and camera permissions
- **Low performance**: Close other browser tabs
- **Detection issues**: Ensure good lighting and face visibility

## License

This project is open source and available under the MIT License.
