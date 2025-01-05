# AI-Powered Adaptive Learning System

## Overview
This project implements an intelligent learning platform that adapts to individual student needs using AI technologies. It combines LSTM and Random Forest algorithms to provide personalized learning experiences and real-time performance tracking.

## Features
- Personalized learning path recommendations
- Real-time performance analytics
- Adaptive assessment system
- Interactive dashboard
- Student progress tracking
- AI-powered predictions

## Technical Stack
- **Backend**: Python, Flask
- **Frontend**: HTML, CSS, JavaScript
- **AI/ML**: TensorFlow, Scikit-learn
- **Data Processing**: NumPy, Pandas
- **Visualization**: Chart.js

## Installation

1. Clone the repository
```bash
git clone https://github.com/yourusername/adaptive-learning-system.git
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Run the application
```bash
python app.py
```

## Project Structure
```
├── app.py                 # Main Flask application
├── templates/
│   └── index.html        # Main dashboard template
├── static/
│   ├── css/             # Stylesheets
│   └── js/              # JavaScript files
└── models/
    ├── ai_model.py      # AI model implementations
    └── analytics.py     # Analytics processing
```

## AI Models

### LSTM Model
- Predicts student performance
- Uses sequence of learning activities
- Features include scores, time spent, and progress

### Random Forest Model
- Recommends learning paths
- Considers prerequisites and difficulty levels
- Evaluates student readiness

## Usage

1. Start the server
2. Access the dashboard at `http://localhost:5000`
3. Select a student to view their personalized dashboard
4. Monitor progress and recommendations in real-time

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

