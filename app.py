from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from collections import defaultdict
import time
from scipy.stats import percentileofscore

app = Flask(__name__)

class AIModel:
    def __init__(self):
        self.performance_predictor = self._create_lstm_predictor()
        self.path_recommender = self._create_path_recommender()
        self.scaler = StandardScaler()
        self._train_models()
        
    def _create_lstm_predictor(self):
       from tensorflow.keras.layers import Input
       from tensorflow.keras.models import Model

       inputs = Input(shape=(10, 5))
       x = LSTM(64, return_sequences=True)(inputs)
       x = Dropout(0.2)(x)
       x = LSTM(32)(x)
       x = Dense(16, activation='relu')(x)
       x = Dropout(0.2)(x)
       outputs = Dense(1, activation='sigmoid')(x)
    
       model = Model(inputs=inputs, outputs=outputs)
       model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
       return model


    def _create_path_recommender(self):
        return RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    
    def _train_models(self):
        X_lstm, y_lstm, X_rf, y_rf = self._generate_training_data()
        self.performance_predictor.fit(X_lstm, y_lstm, epochs=10, batch_size=32, verbose=0)
        self.path_recommender.fit(X_rf, y_rf)

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
                base_score = np.random.normal(75, 15)
                improvement = t * 2
                score = min(max(base_score + improvement, 0), 100)
                
                time_spent = np.random.normal(45, 15)
                knowledge = min(0.1 * t + np.random.normal(0.5, 0.1), 1)
                velocity = score / (time_spent / 60)
                progress = t / n_timesteps
                
                student_sequence.append([
                    score/100,
                    time_spent/60,
                    knowledge,
                    velocity/100,
                    progress
                ])
            
            final_performance = np.mean([seq[0] for seq in student_sequence[-3:]])
            X_lstm.append(student_sequence)
            y_lstm.append(final_performance)
        
        topics = ['algebra', 'geometry', 'calculus', 'physics', 'chemistry']
        difficulties = [0.7, 0.6, 0.9, 0.8, 0.7]
        
        for _ in range(n_students * len(topics)):
            knowledge = np.random.uniform(0, 1)
            topic_idx = np.random.randint(0, len(topics))
            difficulty = difficulties[topic_idx]
            prerequisites_met = np.random.uniform(0, 1)
            past_performance = np.random.normal(0.75, 0.15)
            study_time = np.random.uniform(0.5, 2.0)
            
            success = int(
                knowledge > difficulty - 0.2 and
                prerequisites_met > 0.7 and
                past_performance > 0.6 and
                study_time > 0.8
            )
            
            X_rf.append([
                knowledge,
                difficulty,
                prerequisites_met,
                past_performance,
                study_time
            ])
            y_rf.append(success)
        
        return np.array(X_lstm), np.array(y_lstm), np.array(X_rf), np.array(y_rf)


    #Training LSTM with actual student data ( Not being used here)
    def train_with_student_data(self, student):
        # Prepare sequence data
        X = []
        y = []
        
        for subject, data in student.subjects.items():
            scores = data['scores']
            times = data['time_spent']
            knowledge_states = list(student.knowledge_state.values())
            
            # Create sequences of 10 timesteps
            for i in range(len(scores) - 10):
                sequence = []
                for j in range(10):
                    sequence.append([
                        scores[i+j]/100,  # Normalized score
                        times[i+j]/3600,  # Time in hours
                        knowledge_states[i+j],  # Knowledge state
                        data['recent_velocity'][i+j],  # Learning velocity
                        len(data['topics_completed'])/len(self.topics[subject])  # Progress
                    ])
                
                # Target is the next score
                target = scores[i+10]/100
                
                X.append(sequence)
                y.append(target)
        
        X = np.array(X)
        y = np.array(y)
        
        # Train the model
        self.performance_predictor.fit(
            X, y,
            epochs=10,
            batch_size=32,
            validation_split=0.2,
            verbose=1
        )

    #Update the model ( Not being used here )
    def update_model(self, all_students):
        X_all = []
        y_all = []
        
        for student in all_students:
            X_student, y_student = self.prepare_student_data(student)
            X_all.extend(X_student)
            y_all.extend(y_student)
        
        X_all = np.array(X_all)
        y_all = np.array(y_all)
        
        # Retrain model with all accumulated data
        self.performance_predictor.fit(
            X_all, y_all,
            epochs=10,
            batch_size=32,
            validation_split=0.2,
            verbose=1
        )

    def _prepare_student_sequence(self, student):
        sequence = np.zeros((10, 5))  # Pre-allocate fixed-size array
        current_pos = 0
        
        for subject, data in student.subjects.items():
            scores = data['scores'][-10:] if data['scores'] else []
            if not scores:
                continue
                
            times = [data['time_spent']/len(data['scores']) for _ in range(len(scores))]
            knowledge = [student.knowledge_state.get(topic, 0) for topic in data['topics_completed']][-10:]
            velocities = data['recent_velocity'][-10:] if data['recent_velocity'] else []
            
            max_velocity = max(velocities) if velocities and max(velocities) > 0 else 1
            normalized_velocities = np.array(velocities) / max_velocity if velocities else []
            
            for i in range(min(len(scores), 10 - current_pos)):
                sequence[current_pos + i] = [
                    scores[i] / 100,
                    times[i] / 3600,
                    np.mean(knowledge) if knowledge else 0,
                    normalized_velocities[i] if i < len(normalized_velocities) else 0,
                    len(data['topics_completed']) / len(self.topics[subject])
                ]
            current_pos += min(len(scores), 10 - current_pos)
            if current_pos >= 10:
                break
                
        return np.array([sequence])


    def predict_performance(self, student):
        sequence = self._prepare_student_sequence(student)
        prediction = self.performance_predictor.predict(sequence)
        return float(prediction.item()) 

    def _prepare_topic_features(self, student, topic_info):
        features = []
        for topic, info in topic_info.items():
            knowledge_level = student.knowledge_state.get(topic, 0)
            prerequisites_met = all(
                student.knowledge_state.get(prereq, 0) >= 0.7 
                for prereq in info['prerequisites']
            )
            
            recent_scores = []
            recent_times = []
            for subject_data in student.subjects.values():
                if subject_data['scores']:
                    recent_scores.extend(subject_data['scores'][-5:])
                    avg_time = subject_data['time_spent'] / len(subject_data['scores'])
                    recent_times.append(avg_time)
            
            past_performance = np.mean(recent_scores)/100 if recent_scores else 0.5
            avg_study_time = np.mean(recent_times)/3600 if recent_times else 0.5
            
            features.append([
                knowledge_level,
                info['difficulty'],
                float(prerequisites_met),
                past_performance,
                avg_study_time
            ])
            
        return np.array(features)
    
    def recommend_topics(self, student, topics):
        recommendations = []
        
        # Process each subject and its topics
        for subject, subject_topics in topics.items():
            current_knowledge = {topic: student.knowledge_state.get(topic, 0) 
                               for topic in subject_topics.keys()}
            
            for topic, topic_info in subject_topics.items():
                # Skip already mastered topics (knowledge > 0.8)
                if current_knowledge[topic] > 0.8:
                    continue
                    
                # Check prerequisites
                prereqs_met = True
                prereq_knowledge = 0
                if topic_info['prerequisites']:
                    prereq_scores = [current_knowledge.get(prereq, 0) 
                                   for prereq in topic_info['prerequisites']]
                    prereq_knowledge = sum(prereq_scores) / len(prereq_scores)
                    prereqs_met = prereq_knowledge >= 0.5
                
                # Calculate readiness score
                base_readiness = 0.7 if prereqs_met else 0.3
                knowledge_factor = current_knowledge.get(topic, 0)
                prereq_factor = prereq_knowledge if topic_info['prerequisites'] else 1.0
                
                readiness = (base_readiness * 0.4 + 
                           knowledge_factor * 0.3 + 
                           prereq_factor * 0.3)
                
                # Add to recommendations with calculated readiness
                recommendations.append({
                    'topic': topic,
                    'subject': subject,
                    'readiness': round(readiness, 2),
                    'difficulty': topic_info['difficulty'],
                    'prerequisites': topic_info['prerequisites']
                })
        
        # Sort by readiness and return top recommendations
        sorted_recommendations = sorted(recommendations, 
                                     key=lambda x: x['readiness'], 
                                     reverse=True)
        
        # Return at least 3 recommendations if available
        return sorted_recommendations[:5]

    def set_topics(self, topics):
       self.topics = topics

    def evaluate_student_readiness(self, student, topic):
      # Find subject containing the topic
      subject_topic = None
      for subject, topics in self.topics.items():
        if topic in topics:
            subject_topic = topics[topic]
            break
    
        if not subject_topic:
         return {'readiness_score': 0, 'prerequisites_met': False}

        features = self._prepare_topic_features(student, {topic: subject_topic})
        readiness_score = float(self.path_recommender.predict_proba(features)[0][1])
    
        return {
         'readiness_score': readiness_score,
         'prerequisites_met': all(
            student.knowledge_state.get(prereq, 0) >= 0.7
            for prereq in subject_topic['prerequisites']
          )
        }
    
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
        self._update_learning_velocity(score, time_spent)

    def _update_learning_velocity(self, score, time_spent):
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

class Analytics:
    def __init__(self, learning_system):
        self.learning_system = learning_system

    def get_student_analytics(self, student_id):
        student = self.learning_system.students.get(student_id)
        if not student:
            return None

        return {
            'performance_trends': self._calculate_performance_trends(student),
            'learning_patterns': self._analyze_learning_patterns(student),
            'topic_mastery': self._analyze_topic_mastery(student),
            'time_analytics': self._analyze_time_patterns(student),
            'comparison_metrics': self._get_peer_comparison(student)
        }

    def _calculate_performance_trends(self, student):
        trends = {}
        for subject, data in student.subjects.items():
            scores = data['scores']
            if len(scores) >= 3:
                trends[subject] = {
                    'trend': np.polyfit(range(len(scores)), scores, 1)[0],
                    'recent_avg': np.mean(scores[-3:]),
                    'overall_avg': np.mean(scores),
                    'improvement_rate': (np.mean(scores[-3:]) - np.mean(scores[:3])) 
                                      / np.mean(scores[:3]) if len(scores) >= 6 else 0
                }
        return trends

    def _analyze_learning_patterns(self, student):
        patterns = {}
        for subject, data in student.subjects.items():
            if data['scores'] and data['time_spent']:
                patterns[subject] = {
                    'efficiency': np.mean(data['scores']) / (data['time_spent'] / 3600),
                    'consistency': np.std(data['scores']),
                    'engagement': len(data['topics_completed']) / len(self.learning_system.topics[subject])
                }
        return patterns

    def _analyze_topic_mastery(self, student):
        mastery = {}
        for subject, topics in self.learning_system.topics.items():
            for topic_name, topic_info in topics.items():
                knowledge = student.knowledge_state.get(topic_name, 0)
                prereq_knowledge = np.mean([
                   student.knowledge_state.get(prereq, 0)
                   for prereq in topic_info['prerequisites']
                ]) if topic_info['prerequisites'] else 1.0

                try:
                   readiness = self.learning_system.ai_model.evaluate_student_readiness(student, topic_name)
                   readiness_score = readiness['readiness_score'] if readiness else 0
                except:
                   readiness_score = 0

                mastery[topic_name] = {
                    'knowledge_level': knowledge,
                    'prerequisite_mastery': prereq_knowledge,
                    'readiness': readiness_score
                }
        return mastery

    def _analyze_time_patterns(self, student):
        time_analysis = {}
        for subject, data in student.subjects.items():
            if data['scores'] and data['time_spent']:
                time_per_topic = data['time_spent'] / len(data['topics_completed']) \
                                if data['topics_completed'] else 0
                time_analysis[subject] = {
                    'avg_time_per_topic': time_per_topic,
                    'total_time': data['time_spent'],
                    'efficiency_score': np.mean(data['scores']) / max(time_per_topic, 1)
                }
        return time_analysis

    def _get_peer_comparison(self, student):
        peer_metrics = defaultdict(list)
        for peer_id, peer in self.learning_system.students.items():
            if peer_id != student.student_id and peer.grade == student.grade:
                for subject, data in peer.subjects.items():
                    if data['scores']:
                        peer_metrics[subject].append({
                            'avg_score': np.mean(data['scores']),
                            'topics_completed': len(data['topics_completed']),
                            'time_spent': data['time_spent']
                        })

        comparison = {}
        for subject, metrics in peer_metrics.items():
            if metrics:
                peer_avg_score = np.mean([m['avg_score'] for m in metrics])
                peer_avg_topics = np.mean([m['topics_completed'] for m in metrics])
                peer_avg_time = np.mean([m['time_spent'] for m in metrics])

                student_data = student.subjects[subject]
                student_avg_score = np.mean(student_data['scores']) if student_data['scores'] else 0

                comparison[subject] = {
                    'score_percentile': percentileofscore(
                        [m['avg_score'] for m in metrics],
                        student_avg_score
                    ),
                    'progress_percentile': percentileofscore(
                        [m['topics_completed'] for m in metrics],
                        len(student_data['topics_completed'])
                    ),
                    'efficiency_percentile': percentileofscore(
                        [m['time_spent'] / m['topics_completed'] for m in metrics if m['topics_completed']],
                        student_data['time_spent'] / len(student_data['topics_completed'])
                        if student_data['topics_completed'] else 0
                    )
                }
        return comparison

class AdaptiveLearningSystem:
    def __init__(self):
        self._initialize_topics()  # Call this first
        self.ai_model = AIModel()
        self.ai_model.set_topics(self.topics)
        self.students = {}
        self._initialize_sample_data()
        
        
    def _initialize_topics(self):
        self.topics = {
            'math': {
                'algebra': {
                    'difficulty': 0.7,
                    'prerequisites': ['basic_math']
                },
                'geometry': {
                    'difficulty': 0.6,
                    'prerequisites': ['algebra']
                },
                'calculus': {
                    'difficulty': 0.9,
                    'prerequisites': ['algebra', 'geometry']
                }
            },
            'science': {
                'physics': {
                    'difficulty': 0.8,
                    'prerequisites': ['algebra']
                },
                'chemistry': {
                    'difficulty': 0.7,
                    'prerequisites': ['algebra']
                },
                'biology': {
                    'difficulty': 0.6,
                    'prerequisites': []
                }
            }
        }

    def _initialize_sample_data(self):
        sample_students = [
            ('S001', 'John Doe', 10),
            ('S002', 'Jane Smith', 10),
            ('S003', 'Mike Johnson', 11)
        ]
        
        for student_id, name, grade in sample_students:

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

    def update_student_progress(self, student_id, subject, topic, score, time_spent):
        if student_id not in self.students:
            return None
        
        student = self.students[student_id]
        student.subjects[subject]['scores'].append(score)
        student.subjects[subject]['time_spent'] += time_spent
        
        if score >= 80:
            student.subjects[subject]['topics_completed'].add(topic)
        
        student.update_knowledge(topic, score, time_spent)
        
        velocity = score / (time_spent / 3600)
        student.subjects[subject]['recent_velocity'].append(velocity)
        if len(student.subjects[subject]['recent_velocity']) > 10:
            student.subjects[subject]['recent_velocity'] = student.subjects[subject]['recent_velocity'][-10:]
        
        return self.get_student_data(student_id)


    def get_student_data(self, student_id):
        student = self.students.get(student_id)
        if not student:
            return None
            
        recommended_topics = self.ai_model.recommend_topics(student, self.topics)
        
        return {
            'student_info': {
                'id': student.student_id,
                'name': student.name,
                'grade': student.grade
            },
            'progress': {
                'knowledge_state': dict(student.knowledge_state),
                'performance_prediction': self.ai_model.predict_performance(student),
                'recommended_topics': recommended_topics,
                'learning_velocity': self._calculate_learning_velocity(student),
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


    def _calculate_learning_velocity(self, student):
        if not student.performance_history:
            return 0
        recent_velocities = [ph['velocity'] for ph in student.performance_history[-5:]]
        return round(np.mean(recent_velocities), 2)

class PerformanceMonitor:
    def __init__(self, learning_system):
        self.learning_system = learning_system
        self.thresholds = {
            'low_performance': 60,
            'high_performance': 85,
            'time_warning': 7200,
            'velocity_warning': 10
        }

    def check_student_status(self, student_id):
        student = self.learning_system.students.get(student_id)
        if not student:
            return None

        alerts = []
        recommendations = []

        for subject, data in student.subjects.items():
            recent_scores = data['scores'][-3:] if data['scores'] else []
            if recent_scores and np.mean(recent_scores) < self.thresholds['low_performance']:
                alerts.append({
                    'type': 'low_performance',
                    'subject': subject,
                    'message': f"Performance below threshold in {subject}"
                })
                recommendations.append(self._generate_improvement_plan(student, subject))

        for subject, data in student.subjects.items():
            if data['time_spent'] > self.thresholds['time_warning']:
                alerts.append({
                    'type': 'time_warning',
                    'subject': subject,
                    'message': f"Extended time spent on {subject}"
                })

        velocity = self.learning_system.calculate_learning_velocity(student)
        if velocity < self.thresholds['velocity_warning']:
            alerts.append({
                'type': 'velocity_warning',
                'message': "Learning velocity below expected rate"
            })

        return {
            'alerts': alerts,
            'recommendations': recommendations,
            'status_summary': self._generate_status_summary(student)
        }

    def _generate_improvement_plan(self, student, subject):
        weak_topics = [
            topic for topic in self.learning_system.topics[subject]
            if student.knowledge_state.get(topic, 0) < 0.6
        ]

        return {
            'subject': subject,
            'weak_topics': weak_topics,
            'suggested_actions': [
                {
                    'action': 'review_prerequisites',
                    'topics': self.learning_system.topics[subject][topic]['prerequisites']
                }
                for topic in weak_topics
            ],
            'estimated_improvement_time': len(weak_topics) * 3600
        }

    def _generate_status_summary(self, student):
        return {
            'overall_progress': np.mean([
                len(data['topics_completed']) / len(self.learning_system.topics[subject])
                for subject, data in student.subjects.items()
            ]),
            'average_performance': np.mean([
                np.mean(data['scores']) if data['scores'] else 0
                for data in student.subjects.values()
            ]),
            'learning_efficiency': np.mean([
                np.mean(data['scores']) / (data['time_spent'] / 3600)
                if data['scores'] and data['time_spent'] else 0
                for data in student.subjects.values()
            ])
        }

class DataProcessor:
    @staticmethod
    def process_student_submission(raw_data):
        try:
            processed_data = {
                'student_id': str(raw_data['student_id']),
                'subject': str(raw_data['subject']).lower(),
                'topic': str(raw_data['topic']).lower(),
                'score': float(raw_data['score']),
                'time_spent': int(raw_data['time_spent'])
            }

            assert 0 <= processed_data['score'] <= 100, "Score must be between 0 and 100"
            assert processed_data['time_spent'] > 0, "Time spent must be positive"

            return processed_data
        except (KeyError, ValueError, AssertionError) as e:
            raise ValueError(f"Invalid submission data: {str(e)}")

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

@app.route('/api/analytics/<student_id>')
def get_analytics(student_id):
    analytics = Analytics(learning_system)
    data = analytics.get_student_analytics(student_id)
    if not data:
        return jsonify({'error': 'Student not found'}), 404
    return jsonify(data)

@app.route('/api/status/<student_id>')
def get_status(student_id):
    monitor = PerformanceMonitor(learning_system)
    status = monitor.check_student_status(student_id)
    if not status:
        return jsonify({'error': 'Student not found'}), 404
    return jsonify(status)

@app.route('/api/update_progress', methods=['POST'])
def update_progress():
    try:
        data = DataProcessor.process_student_submission(request.json)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

    updated_data = learning_system.update_student_progress(
        data['student_id'],
        data['subject'],
        data['topic'],
        data['score'],
        data['time_spent']
    )
    
    if not updated_data:
        return jsonify({'error': 'Student not found'}), 404
    
    monitor = PerformanceMonitor(learning_system)
    status = monitor.check_student_status(data['student_id'])
    
    return jsonify({
        'student_data': updated_data,
        'status': status
    })

if __name__ == '__main__':
    app.run(debug=True)
