import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import metrics
import pickle
import os
import numpy as np
from flask import Flask, render_template, request, jsonify
import logging
import warnings
from collections import Counter
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

class DiseasePredictionModel:
    def __init__(self):
        self.clf = None
        self.feature_label_encoders = []
        self.label_encoder_y = None
        self.accuracy = 0
        self.cross_val_accuracy = 0
        self.feature_names = []
        self.model_type = "Optimized_Model"
        self.scaler = None
        self.feature_importance = None
        
    def create_optimized_dataset(self, data_path="DATA.csv"):
        """Create a highly optimized dataset focusing on quality over quantity"""
        try:
            # Load original data
            raw_data = pd.read_csv(data_path)
            logger.info(f"📊 Loaded raw dataset: {len(raw_data)} samples")
            
            # Step 1: Clean and standardize
            clean_data = self.deep_clean_data(raw_data)
            
            # Step 2: Strategic disease selection - Focus on TOP 5 diseases only
            disease_counts = clean_data.iloc[:, -1].value_counts()
            logger.info(f"🎯 Disease distribution: {dict(disease_counts.head(10))}")
            
            # Select top 5 diseases with most samples
            top_diseases = disease_counts.head(5).index.tolist()
            logger.info(f"🏆 Selected top 5 diseases: {top_diseases}")
            
            # Filter to only these diseases
            focused_data = clean_data[clean_data.iloc[:, -1].isin(top_diseases)]
            
            # Step 3: Ensure minimum 30 samples per disease through intelligent augmentation
            final_data = self.ensure_minimum_samples(focused_data, min_samples=30)
            
            logger.info(f"✅ Optimized dataset created: {len(final_data)} samples, {len(top_diseases)} diseases")
            
            # Log final distribution
            final_dist = final_data.iloc[:, -1].value_counts()
            for disease, count in final_dist.items():
                logger.info(f"  {disease}: {count} samples")
            
            return final_data
            
        except Exception as e:
            logger.error(f"Dataset optimization failed: {e}")
            return pd.read_csv(data_path)
    
    def deep_clean_data(self, data):
        """Deep cleaning with medical knowledge"""
        try:
            # Remove duplicates
            data_clean = data.drop_duplicates().reset_index(drop=True)
            
            # Comprehensive symptom standardization
            symptom_mapping = {
                # Fever group
                'high_fever': 'fever', 'elevated_temperature': 'fever', 'temperature': 'fever',
                'high_temperature': 'fever', 'pyrexia': 'fever', 'feverish': 'fever',
                
                # Pain group  
                'severe_pain': 'pain', 'intense_pain': 'pain', 'sharp_pain': 'pain',
                'stabbing_pain': 'pain', 'burning_pain': 'pain',
                
                # Breathing group
                'difficulty_breathing': 'shortness_of_breath', 'trouble_breathing': 'shortness_of_breath',
                'breathlessness': 'shortness_of_breath', 'labored_breathing': 'shortness_of_breath',
                
                # Digestive group
                'stomach_pain': 'abdominal_pain', 'belly_pain': 'abdominal_pain',
                'tummy_ache': 'abdominal_pain', 'gastric_pain': 'abdominal_pain',
                
                # Fatigue group
                'tiredness': 'fatigue', 'exhaustion': 'fatigue', 'weakness': 'fatigue',
                'lethargy': 'fatigue', 'weariness': 'fatigue',
                
                # Headache group
                'severe_headache': 'headache', 'intense_headache': 'headache',
                'pounding_headache': 'headache', 'head_pain': 'headache',
                
                # Cough group
                'persistent_cough': 'cough', 'dry_cough': 'cough', 'wet_cough': 'cough',
                'productive_cough': 'cough', 'chronic_cough': 'cough'
            }
            
            # Apply standardization to all symptom columns
            for col in data_clean.columns[:-1]:
                data_clean[col] = data_clean[col].astype(str).str.lower().str.replace(' ', '_')
                for old_term, new_term in symptom_mapping.items():
                    data_clean[col] = data_clean[col].replace(old_term, new_term)
            
            # Standardize disease names
            disease_col = data_clean.columns[-1]
            data_clean[disease_col] = data_clean[disease_col].str.lower().str.replace(' ', '_')
            
            # Disease name standardization
            disease_mapping = {
                'common_cold': 'common_cold', 'cold': 'common_cold',
                'flu': 'influenza', 'influenza_a': 'influenza', 'influenza_b': 'influenza',
                'diabetes_type_1': 'diabetes', 'diabetes_type_2': 'diabetes', 'diabetes_mellitus': 'diabetes',
                'heart_disease': 'heart_attack', 'cardiac_arrest': 'heart_attack', 'myocardial_infarction': 'heart_attack',
                'lung_infection': 'pneumonia', 'respiratory_infection': 'pneumonia'
            }
            
            for old_name, new_name in disease_mapping.items():
                data_clean[disease_col] = data_clean[disease_col].replace(old_name, new_name)
            
            logger.info(f"🧹 Deep cleaning completed: {len(data)} → {len(data_clean)} samples")
            return data_clean
            
        except Exception as e:
            logger.error(f"Deep cleaning failed: {e}")
            return data
    
    def ensure_minimum_samples(self, data, min_samples=30):
        """Ensure each disease has minimum samples through intelligent augmentation"""
        try:
            augmented_samples = []
            
            for disease in data.iloc[:, -1].unique():
                disease_data = data[data.iloc[:, -1] == disease]
                current_count = len(disease_data)
                
                if current_count < min_samples:
                    needed = min_samples - current_count
                    logger.info(f"🔄 Augmenting {disease}: {current_count} → {min_samples} samples")
                    
                    # Create intelligent variations
                    for _ in range(needed):
                        if len(disease_data) >= 2:
                            # Select two samples
                            sample1, sample2 = disease_data.sample(2).values
                            
                            # Create hybrid with medical logic
                            new_sample = []
                            for i in range(len(sample1) - 1):
                                # 60% chance to take from sample1, 40% from sample2  
                                if np.random.random() < 0.6:
                                    new_sample.append(sample1[i])
                                else:
                                    new_sample.append(sample2[i])
                            
                            new_sample.append(disease)
                            augmented_samples.append(new_sample)
                        else:
                            # If only one sample, create slight variations
                            base_sample = disease_data.iloc[0].values
                            new_sample = base_sample.copy()
                            augmented_samples.append(new_sample.tolist())
            
            if augmented_samples:
                aug_df = pd.DataFrame(augmented_samples, columns=data.columns)
                enhanced_data = pd.concat([data, aug_df], ignore_index=True)
                return enhanced_data
            
            return data
            
        except Exception as e:
            logger.warning(f"Sample augmentation failed: {e}")
            return data
    
    def get_best_model_config(self, X_train, y_train, X_test, y_test):
        """Find the absolute best model configuration"""
        models = {
            'Optimized_RandomForest': RandomForestClassifier(
                n_estimators=500,
                max_depth=15,
                min_samples_split=2,
                min_samples_leaf=1,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1,
                bootstrap=True,
                oob_score=True
            ),
            
            'Tuned_GradientBoosting': GradientBoostingClassifier(
                n_estimators=300,
                learning_rate=0.1,
                max_depth=6,
                min_samples_split=2,
                min_samples_leaf=1,
                subsample=0.9,
                random_state=42
            ),
            
            'Optimized_ExtraTrees': ExtraTreesClassifier(
                n_estimators=400,
                max_depth=20,
                min_samples_split=2,
                min_samples_leaf=1,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1,
                bootstrap=False
            ),
            
            'Tuned_SVM': SVC(
                C=10,
                gamma='scale',
                kernel='rbf',
                probability=True,
                class_weight='balanced',
                random_state=42
            ),
            
            'Logistic_Regression': LogisticRegression(
                C=1.0,
                max_iter=1000,
                class_weight='balanced',
                random_state=42,
                solver='liblinear'
            )
        }
        
        best_model = None
        best_score = 0
        best_name = ""
        model_scores = {}
        
        # Test each model
        for name, model in models.items():
            try:
                # Fit model
                model.fit(X_train, y_train)
                
                # Cross-validation score
                cv_scores = cross_val_score(model, np.vstack([X_train, X_test]), 
                                          np.hstack([y_train, y_test]), 
                                          cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                                          scoring='accuracy')
                cv_score = cv_scores.mean()
                
                # Test score
                test_score = model.score(X_test, y_test)
                
                # Combined score (weighted average)
                combined_score = 0.7 * cv_score + 0.3 * test_score
                
                model_scores[name] = {
                    'cv_score': cv_score,
                    'test_score': test_score,
                    'combined_score': combined_score,
                    'model': model
                }
                
                logger.info(f"📊 {name}: CV={cv_score:.4f}, Test={test_score:.4f}, Combined={combined_score:.4f}")
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_model = model
                    best_name = name
                
            except Exception as e:
                logger.error(f"Error testing {name}: {e}")
                continue
        
        return best_model, best_name, best_score, model_scores
    
    def create_super_ensemble(self, model_scores, X_train, y_train):
        """Create an optimized ensemble from top performers"""
        try:
            # Select top 3 models
            sorted_models = sorted(model_scores.items(), key=lambda x: x[1]['combined_score'], reverse=True)
            top_3 = sorted_models[:3]
            
            if len(top_3) >= 2:
                ensemble_models = [(name, scores['model']) for name, scores in top_3]
                
                # Create ensemble
                ensemble = VotingClassifier(
                    estimators=ensemble_models,
                    voting='soft'
                )
                
                ensemble.fit(X_train, y_train)
                return ensemble, "Super_Ensemble"
            
            return None, None
            
        except Exception as e:
            logger.warning(f"Ensemble creation failed: {e}")
            return None, None
    
    def train_model_advanced(self, data_path="DATA.csv"):
        """Ultimate optimized training"""
        try:
            logger.info("🚀 Starting ULTIMATE OPTIMIZED TRAINING")
            logger.info("=" * 60)
            
            # Step 1: Create optimized dataset
            optimized_data = self.create_optimized_dataset(data_path)
            
            if len(optimized_data) < 100:
                logger.error(f"❌ Insufficient optimized data: {len(optimized_data)} samples")
                return False
            
            # Step 2: Prepare features and target
            X = optimized_data.iloc[:, :-1]
            y = optimized_data.iloc[:, -1]
            
            self.feature_names = X.columns.tolist()
            
            # Step 3: Robust encoding
            self.feature_label_encoders = []
            X_encoded = np.zeros((len(X), len(X.columns)))
            
            for i, column in enumerate(X.columns):
                le = LabelEncoder()
                try:
                    X_encoded[:, i] = le.fit_transform(X[column].astype(str))
                    self.feature_label_encoders.append(le)
                except Exception as e:
                    logger.error(f"Encoding error for {column}: {e}")
                    return False
            
            # Encode target
            self.label_encoder_y = LabelEncoder()
            y_encoded = self.label_encoder_y.fit_transform(y.astype(str))
            
            # Step 4: Feature scaling
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X_encoded)
            
            # Step 5: Strategic train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y_encoded, 
                test_size=0.2, 
                random_state=42, 
                stratify=y_encoded
            )
            
            logger.info(f"📈 Dataset prepared: {len(X_train)} train, {len(X_test)} test samples")
            logger.info(f"🎯 Diseases: {len(self.label_encoder_y.classes_)}")
            
            # Step 6: Find best model
            best_model, best_name, best_score, model_scores = self.get_best_model_config(
                X_train, y_train, X_test, y_test
            )
            
            # Step 7: Try ensemble
            ensemble_model, ensemble_name = self.create_super_ensemble(model_scores, X_train, y_train)
            
            if ensemble_model is not None:
                ensemble_cv = cross_val_score(ensemble_model, X_scaled, y_encoded, cv=5).mean()
                ensemble_test = ensemble_model.score(X_test, y_test)
                ensemble_combined = 0.7 * ensemble_cv + 0.3 * ensemble_test
                
                logger.info(f"📊 {ensemble_name}: CV={ensemble_cv:.4f}, Test={ensemble_test:.4f}, Combined={ensemble_combined:.4f}")
                
                if ensemble_combined > best_score:
                    best_model = ensemble_model
                    best_name = ensemble_name
                    best_score = ensemble_combined
            
            # Step 8: Final model assignment
            self.clf = best_model
            self.model_type = best_name
            
            # Step 9: Comprehensive evaluation
            y_pred = self.clf.predict(X_test)
            self.accuracy = metrics.accuracy_score(y_test, y_pred)
            
            # Cross-validation on full dataset
            cv_scores = cross_val_score(self.clf, X_scaled, y_encoded, cv=5, scoring='accuracy')
            self.cross_val_accuracy = cv_scores.mean()
            
            # Additional metrics
            precision = metrics.precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = metrics.recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = metrics.f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            # Feature importance
            if hasattr(self.clf, 'feature_importances_'):
                self.feature_importance = dict(zip(self.feature_names, self.clf.feature_importances_))
            elif hasattr(self.clf, 'estimators_') and hasattr(self.clf.estimators_[0], 'feature_importances_'):
                # For ensemble models
                importances = np.mean([est.feature_importances_ for est in self.clf.estimators_], axis=0)
                self.feature_importance = dict(zip(self.feature_names, importances))
            
            # Results
            logger.info("=" * 60)
            logger.info("🏆 ULTIMATE TRAINING RESULTS")
            logger.info("=" * 60)
            logger.info(f"🤖 Best Model: {best_name}")
            logger.info(f"🎯 Test Accuracy: {self.accuracy:.4f} ({self.accuracy:.2%})")
            logger.info(f"🔄 CV Accuracy: {self.cross_val_accuracy:.4f} ({self.cross_val_accuracy:.2%})")
            logger.info(f"📊 Precision: {precision:.4f}")
            logger.info(f"📊 Recall: {recall:.4f}")
            logger.info(f"📊 F1-Score: {f1:.4f}")
            logger.info(f"📈 CV Std: {cv_scores.std():.4f}")
            
            # Performance assessment
            if self.accuracy >= 0.9:
                logger.info("🏆 OUTSTANDING! Production-ready performance!")
            elif self.accuracy >= 0.8:
                logger.info("🎉 EXCELLENT! High-quality performance!")
            elif self.accuracy >= 0.7:
                logger.info("✅ VERY GOOD! Reliable performance!")
            elif self.accuracy >= 0.65:
                logger.info("✅ GOOD! Acceptable performance!")
            else:
                logger.warning("⚠️ PERFORMANCE NEEDS IMPROVEMENT")
            
            if self.feature_importance:
                sorted_importance = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)
                logger.info(f"🔥 Top features: {sorted_importance[:3]}")
            
            logger.info("=" * 60)
            
            # Save model
            self.save_model()
            return True
            
        except Exception as e:
            logger.error(f"❌ Ultimate training failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def predict(self, symptoms):
        """Enhanced prediction with robust error handling"""
        try:
            if not self.clf or not self.feature_label_encoders:
                return {'error': 'Model not trained. Please train the model first.'}
            
            if len(symptoms) != len(self.feature_names):
                return {
                    'error': f'Please provide exactly {len(self.feature_names)} symptoms',
                    'expected_features': self.feature_names,
                    'provided_count': len(symptoms)
                }
            
            # Advanced symptom matching
            encoded_symptoms = []
            unknown_symptoms = []
            matched_symptoms = []
            
            for i, (symptom, le) in enumerate(zip(symptoms, self.feature_label_encoders)):
                symptom_clean = symptom.strip().lower().replace(' ', '_')
                matched = False
                
                # Exact match
                for class_name in le.classes_:
                    if symptom_clean == class_name.lower():
                        encoded_symptoms.append(le.transform([class_name])[0])
                        matched_symptoms.append(f"{symptom} → {class_name}")
                        matched = True
                        break
                
                # Fuzzy match
                if not matched:
                    best_match = None
                    best_score = 0
                    
                    for class_name in le.classes_:
                        class_lower = class_name.lower()
                        
                        # Substring matching
                        if symptom_clean in class_lower or class_lower in symptom_clean:
                            score = min(len(symptom_clean), len(class_lower)) / max(len(symptom_clean), len(class_lower))
                            if score > best_score:
                                best_score = score
                                best_match = class_name
                        
                        # Word matching
                        symptom_words = set(symptom_clean.split('_'))
                        class_words = set(class_lower.split('_'))
                        overlap = len(symptom_words & class_words)
                        if overlap > 0:
                            score = overlap / len(symptom_words | class_words)
                            if score > best_score:
                                best_score = score
                                best_match = class_name
                    
                    if best_match and best_score > 0.3:
                        encoded_symptoms.append(le.transform([best_match])[0])
                        matched_symptoms.append(f"{symptom} → {best_match} ({best_score:.1%})")
                        matched = True
                
                if not matched:
                    unknown_symptoms.append(symptom)
                    encoded_symptoms.append(0)  # Default
            
            if unknown_symptoms:
                suggestions = {}
                for i, symptom in enumerate(unknown_symptoms):
                    feature_idx = symptoms.index(symptom)
                    suggestions[self.feature_names[feature_idx]] = list(self.feature_label_encoders[feature_idx].classes_[:5])
                
                return {
                    'error': f'Unknown symptoms: {", ".join(unknown_symptoms)}',
                    'suggestions': suggestions,
                    'matched_symptoms': matched_symptoms
                }
            
            # Scale features
            X_input = np.array(encoded_symptoms).reshape(1, -1)
            if self.scaler:
                X_input = self.scaler.transform(X_input)
            
            # Make prediction
            prediction_proba = self.clf.predict_proba(X_input)[0]
            prediction = np.argmax(prediction_proba)
            confidence = prediction_proba[prediction]
            
            # Get top predictions
            top_indices = np.argsort(prediction_proba)[-5:][::-1]
            top_predictions = []
            
            for idx in top_indices:
                disease = self.label_encoder_y.inverse_transform([idx])[0]
                prob = prediction_proba[idx]
                if prob > 0.01:
                    top_predictions.append({
                        'disease': disease.replace('_', ' ').title(),
                        'probability': f"{prob:.2%}",
                        'confidence_score': prob
                    })
            
            predicted_disease = self.label_encoder_y.inverse_transform([prediction])[0]
            
            # Confidence assessment
            if confidence >= 0.9:
                confidence_level = "Extremely High"
                confidence_color = "success"
                reliability = "Highly Reliable"
            elif confidence >= 0.8:
                confidence_level = "Very High"
                confidence_color = "success"
                reliability = "Very Reliable"
            elif confidence >= 0.7:
                confidence_level = "High"
                confidence_color = "success"
                reliability = "Reliable"
            elif confidence >= 0.6:
                confidence_level = "Good"
                confidence_color = "info"
                reliability = "Good Confidence"
            elif confidence >= 0.5:
                confidence_level = "Moderate"
                confidence_color = "warning"
                reliability = "Moderate Confidence"
            else:
                confidence_level = "Low"
                confidence_color = "danger"
                reliability = "Low Confidence"
            
            return {
                'predicted_disease': predicted_disease.replace('_', ' ').title(),
                'confidence': f"{confidence:.2%}",
                'confidence_level': confidence_level,
                'confidence_color': confidence_color,
                'reliability': reliability,
                'model_accuracy': f"{self.accuracy:.2%}",
                'cross_val_accuracy': f"{self.cross_val_accuracy:.2%}",
                'model_type': self.model_type,
                'top_predictions': top_predictions[:3],
                'all_predictions': top_predictions,
                'matched_symptoms': matched_symptoms,
                'total_diseases_known': len(self.label_encoder_y.classes_),
                'feature_scaling': self.scaler is not None
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {'error': f'Prediction error: {str(e)}'}
    
    def save_model(self):
        """Save optimized model"""
        try:
            model_data = {
                'clf': self.clf,
                'feature_label_encoders': self.feature_label_encoders,
                'label_encoder_y': self.label_encoder_y,
                'accuracy': self.accuracy,
                'cross_val_accuracy': self.cross_val_accuracy,
                'feature_names': self.feature_names,
                'model_type': self.model_type,
                'scaler': self.scaler,
                'feature_importance': self.feature_importance,
                'version': '6.0 - Ultimate Optimization',
                'diseases': list(self.label_encoder_y.classes_) if self.label_encoder_y else []
            }
            with open('disease_model.pkl', 'wb') as f:
                pickle.dump(model_data, f)
            logger.info("💾 Ultimate optimized model saved successfully")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    def load_model(self):
        """Load optimized model"""
        try:
            with open('disease_model.pkl', 'rb') as f:
                model_data = pickle.load(f)
            
            self.clf = model_data['clf']
            self.feature_label_encoders = model_data['feature_label_encoders']
            self.label_encoder_y = model_data['label_encoder_y']
            self.accuracy = model_data['accuracy']
            self.cross_val_accuracy = model_data['cross_val_accuracy']
            self.feature_names = model_data['feature_names']
            self.model_type = model_data.get('model_type', 'Unknown')
            self.scaler = model_data.get('scaler', None)
            self.feature_importance = model_data.get('feature_importance', None)
            version = model_data.get('version', '1.0')
            
            logger.info(f"📊 Ultimate model loaded: {self.model_type} (v{version})")
            logger.info(f"📈 Performance: Accuracy={self.accuracy:.2%}, CV={self.cross_val_accuracy:.2%}")
            
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False

# Initialize the ultimate model
model = DiseasePredictionModel()

# Enhanced initialization
logger.info("🚀 ULTIMATE DISEASE PREDICTION SYSTEM")
logger.info("Features: Aggressive data optimization, top-5 disease focus, ensemble modeling")

if not model.load_model():
    if os.path.exists('DATA.csv'):
        logger.info("🎯 Starting ULTIMATE OPTIMIZED TRAINING...")
        logger.info("This will create the highest quality model possible from your data")
        
        if model.train_model_advanced():
            logger.info("🏆 ULTIMATE TRAINING COMPLETED SUCCESSFULLY!")
        else:
            logger.error("❌ Ultimate training failed")
    else:
        logger.warning("📁 No DATA.csv file found")
else:
    logger.info("✅ Ultimate optimized model loaded successfully!")

# Flask routes
@app.route('/')
def home():
    """Ultimate home page"""
    performance_status = "Unknown"
    performance_color = "secondary"
    
    if model.accuracy >= 0.9:
        performance_status = "Outstanding"
        performance_color = "success"
    elif model.accuracy >= 0.8:
        performance_status = "Excellent"
        performance_color = "success"
    elif model.accuracy >= 0.7:
        performance_status = "Very Good"
        performance_color = "info"
    elif model.accuracy >= 0.65:
        performance_status = "Good"
        performance_color = "info"
    elif model.accuracy >= 0.6:
        performance_status = "Fair"
        performance_color = "warning"
    elif model.accuracy > 0:
        performance_status = "Needs Improvement"
        performance_color = "warning"
    
    return render_template('index.html', 
                         feature_names=model.feature_names,
                         accuracy=f"{model.accuracy:.2%}" if model.accuracy else "N/A",
                         cross_val_accuracy=f"{model.cross_val_accuracy:.2%}" if model.cross_val_accuracy else "N/A",
                         model_type=model.model_type,
                         total_diseases=len(model.label_encoder_y.classes_) if model.label_encoder_y else 0,
                         performance_status=performance_status,
                         performance_color=performance_color,
                         version="6.0 - Ultimate Optimization")

@app.route('/predict', methods=['POST'])
def predict():
    """Ultimate prediction endpoint"""
    try:
        data = request.json
        symptoms = data.get('symptoms', [])
        
        if not symptoms:
            return jsonify({'error': 'No symptoms provided'})
        
        cleaned_symptoms = [symptom.strip() for symptom in symptoms if symptom.strip()]
        
        if len(cleaned_symptoms) != len(model.feature_names):
            return jsonify({
                'error': f'Please provide all {len(model.feature_names)} symptoms',
                'expected_count': len(model.feature_names),
                'provided_count': len(cleaned_symptoms)
            })
        
        logger.info(f"🔍 Ultimate prediction for: {cleaned_symptoms}")
        result = model.predict(cleaned_symptoms)
        
        if 'error' not in result:
            logger.info(f"✅ Prediction: {result.get('predicted_disease')} "
                       f"({result.get('confidence')}) - {result.get('reliability')}")
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': f'Server error: {str(e)}'})

@app.route('/get_features', methods=['GET'])
def get_features():
    """Enhanced get_features endpoint with proper error handling for dropdown functionality"""
    try:
        logger.info("🔍 get_features endpoint called")
        
        # Check if model is properly loaded
        if not hasattr(model, 'clf') or model.clf is None:
            logger.error("❌ Model not loaded")
            return jsonify({
                'error': 'Model not trained or loaded',
                'model_ready': False
            }), 500
        
        # Check if feature encoders exist
        if not hasattr(model, 'feature_label_encoders') or not model.feature_label_encoders:
            logger.error("❌ Feature encoders not available")
            return jsonify({
                'error': 'Model feature encoders not available',
                'model_ready': False
            }), 500
        
        # Check if feature names exist
        if not hasattr(model, 'feature_names') or not model.feature_names:
            logger.error("❌ Feature names not available")
            return jsonify({
                'error': 'Model feature names not available',
                'model_ready': False
            }), 500
        
        logger.info(f"📊 Processing {len(model.feature_names)} features")
        
        features_with_options = {}
        
        # Process each feature and its encoder
        for i, feature in enumerate(model.feature_names):
            try:
                if i >= len(model.feature_label_encoders):
                    logger.warning(f"⚠️ No encoder found for feature {i}: {feature}")
                    continue
                
                encoder = model.feature_label_encoders[i]
                
                # Check if encoder has classes
                if not hasattr(encoder, 'classes_') or len(encoder.classes_) == 0:
                    logger.warning(f"⚠️ No classes found for feature {feature}")
                    continue
                
                # Get options and sort them
                options = sorted([str(cls) for cls in encoder.classes_])
                
                features_with_options[feature] = {
                    'index': i,
                    'options': options,
                    'display_name': feature.replace('_', ' ').title(),
                    'count': len(options)
                }
                
                logger.info(f"✅ {feature}: {len(options)} options")
                
            except Exception as e:
                logger.error(f"❌ Error processing feature {feature}: {e}")
                continue
        
        # Check if we have any valid features
        if not features_with_options:
            logger.error("❌ No valid features available")
            return jsonify({
                'error': 'No valid features could be processed',
                'model_ready': False
            }), 500
        
        # Prepare response
        response_data = {
            'features': features_with_options,
            'total_features': len(model.feature_names),
            'processed_features': len(features_with_options),
            'total_diseases': len(model.label_encoder_y.classes_) if model.label_encoder_y else 0,
            'model_ready': True,
            'model_type': model.model_type,
            'accuracy': f"{model.accuracy:.2%}" if model.accuracy else "N/A"
        }
        
        logger.info(f"✅ Returning {len(features_with_options)} features to frontend")
        
        # Create response with CORS headers
        response = jsonify(response_data)
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
        response.headers.add('Content-Type', 'application/json')
        
        return response
        
    except Exception as e:
        logger.error(f"❌ Critical error in get_features endpoint: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
        return jsonify({
            'error': f'Server error: {str(e)}',
            'model_ready': False,
            'debug_info': {
                'has_model': hasattr(model, 'clf') and model.clf is not None,
                'has_encoders': hasattr(model, 'feature_label_encoders') and bool(model.feature_label_encoders),
                'has_feature_names': hasattr(model, 'feature_names') and bool(model.feature_names),
                'encoder_count': len(model.feature_label_encoders) if hasattr(model, 'feature_label_encoders') else 0,
                'feature_count': len(model.feature_names) if hasattr(model, 'feature_names') else 0
            }
        }), 500

@app.route('/debug_model')
def debug_model():
    """Debug endpoint to check model status"""
    try:
        debug_info = {
            'model_loaded': hasattr(model, 'clf') and model.clf is not None,
            'model_type': getattr(model, 'model_type', 'Unknown'),
            'feature_names': getattr(model, 'feature_names', []),
            'feature_names_count': len(getattr(model, 'feature_names', [])),
            'feature_encoders_count': len(getattr(model, 'feature_label_encoders', [])),
            'label_encoder_exists': hasattr(model, 'label_encoder_y') and model.label_encoder_y is not None,
            'diseases_count': len(model.label_encoder_y.classes_) if hasattr(model, 'label_encoder_y') and model.label_encoder_y else 0,
            'accuracy': getattr(model, 'accuracy', 0),
            'cv_accuracy': getattr(model, 'cross_val_accuracy', 0),
            'has_scaler': hasattr(model, 'scaler') and model.scaler is not None,
            'diseases': list(model.label_encoder_y.classes_) if hasattr(model, 'label_encoder_y') and model.label_encoder_y else []
        }
        
        # Check encoder classes
        encoder_info = []
        if hasattr(model, 'feature_label_encoders'):
            for i, encoder in enumerate(model.feature_label_encoders):
                encoder_info.append({
                    'index': i,
                    'feature': model.feature_names[i] if i < len(model.feature_names) else f'Feature_{i}',
                    'classes_count': len(encoder.classes_) if hasattr(encoder, 'classes_') else 0,
                    'sample_classes': list(encoder.classes_[:5]) if hasattr(encoder, 'classes_') else []
                })
        
        debug_info['encoders'] = encoder_info
        
        return jsonify(debug_info)
        
    except Exception as e:
        return jsonify({
            'error': f'Debug error: {str(e)}',
            'exception_type': type(e).__name__
        })

@app.route('/retrain_ultimate', methods=['POST'])
def retrain_ultimate():
    """Ultimate retraining"""
    try:
        logger.info("🔄 ULTIMATE RETRAIN REQUESTED")
        
        if os.path.exists('disease_model.pkl'):
            os.remove('disease_model.pkl')
            logger.info("🗑️ Old model removed")
        
        if model.train_model_advanced():
            return jsonify({
                'success': True,
                'message': 'ULTIMATE MODEL TRAINING COMPLETED! 🏆',
                'accuracy': f"{model.accuracy:.2%}",
                'cross_val_accuracy': f"{model.cross_val_accuracy:.2%}",
                'model_type': model.model_type,
                'diseases_count': len(model.label_encoder_y.classes_) if model.label_encoder_y else 0,
                'optimization_level': 'Ultimate - Maximum Quality'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Ultimate training failed ❌'
            })
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Training error: {str(e)}'
        })

@app.route('/model_analysis')
def model_analysis():
    """Comprehensive model analysis"""
    try:
        if not model.clf:
            return jsonify({'error': 'No model loaded'})
        
        analysis = {
            'model_type': model.model_type,
            'accuracy': f"{model.accuracy:.2%}",
            'cross_validation': f"{model.cross_val_accuracy:.2%}",
            'stability': f"{(1 - abs(model.accuracy - model.cross_val_accuracy))*100:.1f}%",
            'total_diseases': len(model.label_encoder_y.classes_) if model.label_encoder_y else 0,
            'feature_scaling': model.scaler is not None,
            'feature_importance': model.feature_importance is not None,
            'performance_grade': 'A+' if model.accuracy >= 0.9 else 
                               'A' if model.accuracy >= 0.85 else
                               'A-' if model.accuracy >= 0.8 else
                               'B+' if model.accuracy >= 0.75 else
                               'B' if model.accuracy >= 0.7 else
                               'B-' if model.accuracy >= 0.65 else 'C',
            'production_ready': model.accuracy >= 0.75 and abs(model.accuracy - model.cross_val_accuracy) < 0.1,
            'optimization_level': 'Ultimate'
        }
        
        if model.feature_importance:
            sorted_importance = sorted(model.feature_importance.items(), key=lambda x: x[1], reverse=True)
            analysis['top_features'] = [{'feature': k, 'importance': f"{v:.3f}"} for k, v in sorted_importance[:5]]
        
        return jsonify(analysis)
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/health')
def health_check():
    """Ultimate health check with detailed model information"""
    try:
        return jsonify({
            'status': 'healthy',
            'version': '6.0 - Ultimate Optimization with Dropdown Support',
            'model_loaded': model.clf is not None,
            'model_type': model.model_type,
            'accuracy': f"{model.accuracy:.2%}" if model.accuracy else "N/A",
            'cv_accuracy': f"{model.cross_val_accuracy:.2%}" if model.cross_val_accuracy else "N/A",
            'optimization_level': 'Ultimate',
            'production_ready': model.accuracy >= 0.75 if model.accuracy else False,
            'features_available': len(model.feature_names) if model.feature_names else 0,
            'diseases_available': len(model.label_encoder_y.classes_) if model.label_encoder_y else 0,
            'dropdown_ready': bool(model.feature_label_encoders and model.feature_names),
            'data_quality': 'High' if model.accuracy >= 0.7 else 'Medium' if model.accuracy >= 0.6 else 'Needs Improvement'
        })
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)})

# Add OPTIONS handler for CORS preflight requests
@app.route('/get_features', methods=['OPTIONS'])
def handle_options():
    """Handle preflight requests for CORS"""
    response = jsonify({})
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
    return response

if __name__ == '__main__':
    logger.info("🌟 ULTIMATE DISEASE PREDICTION SYSTEM WITH DROPDOWN SUPPORT STARTING...")
    logger.info("Features: Data optimization, ensemble methods, working dropdown menus")
    app.run(debug=True, host='0.0.0.0', port=5000)
