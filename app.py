from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from collections import defaultdict
import time
import tensorflow as tf

app = Flask(__name__)

class AIModel:
    def __init__(self):
        self.performance_predictor = self._create_lstm_predictor()
        self.path_recommender = self._create_path_recommender()
        self.scaler = StandardScaler()
        self._train_models()
        
    def _generate_training_data(self):
        n_students = 100
        n_timesteps = 10
        n_features = 5
        
        X_lstm = []
        y_lstm = []
        X_rf = []
        y_rf = []
        
        for _ in range(n_students):
            student_sequence = []
            for t in range(n_timesteps):
                score = np.random.normal(75, 15)
                time_spent = np.random.normal(45, 15)
                difficulty = np.random.uniform(0.3, 0.9)
                prior_knowledge = np.random.uniform(0, 1)
                attempts = np.random.randint(1, 4)
                
                student_sequence.append([
                    score/100, time_spent/60, difficulty,
                    prior_knowledge, attempts/3
                ])
            
            final_performance = np.random.uniform(0, 1)
            X_lstm.append(student_sequence)
            y_lstm.append(final_performance)
            
            for topic in ['algebra', 'geometry', 'calculus', 'physics']:
                features = [
                    np.random.uniform(0, 1),  # knowledge_level
                    np.random.uniform(0.3, 0.9),  # topic_difficulty
                    np.random.uniform(0, 1),  # prerequisites_met
                    np.random.normal(0.75, 0.15),  # past_performance
                    np.random.uniform(0, 1)  # time_available
                ]
                
                success = 1 if (features[0] + features[2] > features[1] * 1.5 
                              and features[3] > 0.6) else 0
                
                X_rf.append(features)
                y_rf.append(success)
        
        return np.array(X_lstm), np.array(y_lstm), np.array(X_rf), np.array(y_rf)

    def _create_lstm_predictor(self):
        model = Sequential([
            LSTM(64, input_shape=(10, 5), return_sequences=True),
            Dropout(0.2),
            LSTM(32),
            Dense(16, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model
    
    def _create_path_recommender(self):
        return RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    
    def _train_models(self):
        X_lstm, y_lstm, X_rf, y_rf = self._generate_training_data()
        self.performance_predictor.fit(X_lstm, y_lstm, epochs=10, batch_size=32, verbose=0)
        self.path_recommender.fit(X_rf, y_rf)
    
    def predict_performance(self, student_data):
        sequence = self._prepare_sequence_data(student_data)
        return float(self.performance_predictor.predict(np.array([sequence]))[0])
    
    def _prepare_sequence_data(self, student_data):
        sequence = []
        for subject in student_data['subjects'].values():
            scores = subject.get('scores', [])
            times = subject.get('time_spent', [])
            
            if scores and times:
                recent_scores = scores[-10:]
                avg_time = times / max(len(scores), 1)
                sequence.append([
                    np.mean(recent_scores)/100,
                    avg_time/3600,
                    0.7,  # difficulty
                    0.5,  # knowledge
                    1    # attempts
                ])
        
        while len(sequence) < 10:
            sequence.append([0, 0, 0, 0, 0])
            
        return sequence

    def recommend_next_topic(self, knowledge_level, topic_difficulty, prerequisites_met,
                           past_performance, time_available):
        features = np.array([[
            knowledge_level, topic_difficulty, prerequisites_met,
            past_performance, time_available
        ]])
        return bool(self.path_recommender.predict(features)[0])

class Student:
    def __init__(self, student_id, name, grade):
        self.student_id = student_id
        self.name = name
        self.grade = grade
        self.performance_history = []
        self.knowledge_state = defaultdict(float)
        self.learning_path = []
        self.subjects = defaultdict(lambda: {
            'scores': [],
            'time_spent': 0,
            'topics_completed': set(),
            'mastery_level': defaultdict(float),
            'score_timestamps': [],
            'recent_velocity': []
        })
        self.last_update_time = time.time()

    def update_knowledge(self, topic, score, time_spent):
        current_knowledge = self.knowledge_state[topic]
        performance_weight = score / 100
        time_weight = min(time_spent / 3600, 1)
        
        new_knowledge = current_knowledge + (
            performance_weight * 0.7 + 
            time_weight * 0.3
        ) * (1 - current_knowledge)
        
        self.knowledge_state[topic] = round(new_knowledge, 2)
        self.update_learning_velocity(score, time_spent)

    def update_learning_velocity(self, score, time_spent):
        current_time = time.time()
        time_diff = current_time - self.last_update_time
        
        if time_diff > 0:
            velocity = score / (time_spent / 3600)
            self.performance_history.append({
                'timestamp': current_time,
                'score': score,
                'velocity': velocity
            })
            
            if len(self.performance_history) > 10:
                self.performance_history = self.performance_history[-10:]
        
        self.last_update_time = current_time

class AdaptiveLearningSystem:
    def __init__(self):
        self.ai_model = AIModel()
        self.students = {}
        self.topics = self._initialize_topics()
        self._initialize_sample_data()
        
    def _initialize_topics(self):
        return {
            'math': {
                'algebra': {'difficulty': 0.7, 'prerequisites': ['basic_math']},
                'geometry': {'difficulty': 0.6, 'prerequisites': ['algebra']},
                'calculus': {'difficulty': 0.9, 'prerequisites': ['algebra', 'geometry']},
                'statistics': {'difficulty': 0.8, 'prerequisites': ['algebra']}
            },
            'science': {
                'physics': {'difficulty': 0.8, 'prerequisites': ['algebra']},
                'chemistry': {'difficulty': 0.7, 'prerequisites': ['algebra']},
                'biology': {'difficulty': 0.6, 'prerequisites': []},
                'astronomy': {'difficulty': 0.7, 'prerequisites': ['physics']}
            }
        }

    def _initialize_sample_data(self):
        for student_id, name, grade in [
            ('S001', 'John Doe', 10),
            ('S002', 'Jane Smith', 10),
            ('S003', 'Mike Johnson', 11)
        ]:
            self.students[student_id] = Student(student_id, name, grade)
            self._initialize_student_data(self.students[student_id])

    def _initialize_student_data(self, student):
        for subject in ['math', 'science']:
            for topic in self.topics[subject]:
                score = np.random.randint(70, 100)
                time = np.random.randint(1800, 3600)
                student.update_knowledge(topic, score, time)
                student.subjects[subject]['scores'].append(score)
                student.subjects[subject]['time_spent'] += time
                if score >= 80:
                    student.subjects[subject]['topics_completed'].add(topic)

    def calculate_learning_velocity(self, student):
        if not student.performance_history:
            return 0
        recent_velocities = [ph['velocity'] for ph in student.performance_history[-5:]]
        return round(np.mean(recent_velocities), 2)

    def predict_performance(self, student_id):
        student = self.students[student_id]
        student_data = {
            'subjects': student.subjects
        }
        return self.ai_model.predict_performance(student_data)

    def recommend_topics(self, student_id):
        student = self.students[student_id]
        available_topics = []
        
        for subject, topics in self.topics.items():
            for topic, info in topics.items():
                if topic not in student.learning_path:
                    prerequisites_met = all(
                        student.knowledge_state[prereq] >= 0.7 
                        for prereq in info['prerequisites']
                    )
                    
                    if prerequisites_met:
                        knowledge_level = student.knowledge_state.get(topic, 0)
                        past_performance = np.mean([
                            score/100 for subj in student.subjects.values() 
                            for score in subj['scores']
                        ]) if any(subj['scores'] for subj in student.subjects.values()) else 0.5
                        
                        recommended = self.ai_model.recommend_next_topic(
                            knowledge_level,
                            info['difficulty'],
                            float(prerequisites_met),
                            past_performance,
                            0.8  # default time_available
                        )
                        
                        if recommended:
                            available_topics.append({
                                'topic': topic,
                                'subject': subject,
                                'difficulty': info['difficulty'],
                                'readiness': 1 - info['difficulty'] + knowledge_level
                            })
        
        return sorted(available_topics, key=lambda x: x['readiness'], reverse=True)[:3]

    def update_student_progress(self, student_id, subject, topic, score, time_spent):
        if student_id not in self.students:
            return None
        
        student = self.students[student_id]
        student.subjects[subject]['scores'].append(score)
        student.subjects[subject]['time_spent'] += time_spent
        
        if score >= 80:
            student.subjects[subject]['topics_completed'].add(topic)
        
        student.update_knowledge(topic, score, time_spent)
        
        return self.get_student_data(student_id)

    def get_student_data(self, student_id):
        student = self.students.get(student_id)
        if not student:
            return None
        
        return {
            'student_info': {
                'id': student.student_id,
                'name': student.name,
                'grade': student.grade
            },
            'progress': {
                'knowledge_state': dict(student.knowledge_state),
                'performance_prediction': self.predict_performance(student_id),
                'recommended_topics': self.recommend_topics(student_id),
                'learning_velocity': self.calculate_learning_velocity(student),
                'subjects': {
                    subject: {
                        'average_score': round(np.mean(data['scores']) if data['scores'] else 0, 2),
                        'time_spent': data['time_spent'],
                        'completed_topics': list(data['topics_completed']),
                        'scores': data['scores']
                    }
                    for subject, data in student.subjects.items()
                }
            }
        }

learning_system = AdaptiveLearningSystem()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/student/<student_id>')
def get_student(student_id):
    data = learning_system.get_student_data(student_id)
    if not data:
        return jsonify({'error': 'Student not found'}), 404
    return jsonify(data)

@app.route('/api/update_progress', methods=['POST'])
def update_progress():
    data = request.json
    updated_data = learning_system.update_student_progress(
        data['student_id'],
        data['subject'],
        data['topic'],
        data['score'],
        data['time_spent']
    )
    if not updated_data:
        return jsonify({'error': 'Student not found'}), 404
    return jsonify(updated_data)

if __name__ == '__main__':
    app.run(debug=True)