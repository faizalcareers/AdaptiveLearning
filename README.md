# AI-Powered Adaptive Learning System

An intelligent learning platform that personalizes educational content using LSTM and RandomForest models for performance prediction and path recommendation.

## Features

- Real-time learning analytics
- AI-powered topic recommendations
- Knowledge state tracking
- Learning velocity calculations
- Interactive dashboard
- Multi-subject support

## Tech Stack

- Backend: Python, Flask, TensorFlow, scikit-learn
- Frontend: HTML5, CSS3, JavaScript, Chart.js
- AI Models: LSTM (performance prediction), RandomForest (path recommendation)

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/adaptive-learning-system.git
cd adaptive-learning-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

## Configuration

Create `requirements.txt`:

```text
flask==2.0.1
numpy==1.21.0
pandas==1.3.0
scikit-learn==0.24.2
tensorflow==2.8.0
```

## Usage

1. Start server:
```bash
python app.py
```

2. Access dashboard:
```
http://localhost:5000
```

3. API Endpoints:
```
GET /api/student/<student_id>
POST /api/update_progress
```

## Data Structure

```python
student_data = {
    'student_info': {
        'id': 'S001',
        'name': 'John Doe',
        'grade': 10
    },
    'progress': {
        'knowledge_state': {...},
        'performance_prediction': 0.85,
        'learning_velocity': 25.5
    }
}
```

## AI Model Parameters

```python
lstm_model = Sequential([
    LSTM(64, input_shape=(10, 5)),
    Dense(1, activation='sigmoid')
])

rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10
)
```

## Contributing

1. Fork repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Create Pull Request

## License

MIT License - See LICENSE.md
