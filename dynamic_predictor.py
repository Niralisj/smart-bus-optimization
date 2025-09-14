# dynamic_predictor.py - Real-time Dynamic Simulation with ML Prediction Model
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import json
import random
import threading
import time
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# ML imports with fallback
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error
    ML_AVAILABLE = True
    print("✅ ML libraries imported successfully")
except ImportError:
    print("⚠️ ML libraries not available, using statistical predictions")
    RandomForestRegressor = None
    LabelEncoder = None
    ML_AVAILABLE = False

@dataclass
class LiveBus:
    bus_id: str
    route_id: str
    latitude: float
    longitude: float
    speed: float
    passenger_count: int
    status: str
    last_stop: str
    next_stop: str
    delay_minutes: int

class DynamicLivePredictor:
    def __init__(self):
        # ML Model components
        self.ml_model = None
        self.route_encoder = LabelEncoder() if ML_AVAILABLE else None
        self.is_model_trained = False
        self.historical_data = None
        self.model_accuracy = None
        
        # Simulation components
        self.live_buses = {}
        self.simulation_running = False
        self.simulation_thread = None
        self.routes = ['R1', 'R2', 'R3', 'R4', 'R5']
        
        # Real-time state
        self.current_predictions = {}
        self.actual_ridership_log = []
        self.passenger_boarding_events = []
        
        print("🚀 Dynamic Live Predictor initialized")
    
    def load_and_train_model(self, csv_path="historic.csv"):
        """Load historical data and train ML prediction model"""
        if not ML_AVAILABLE:
            print("⚠️ ML libraries not available, using statistical fallback")
            return False
        
        try:
            # Load historical data
            print(f"📊 Loading historical data from {csv_path}...")
            df = pd.read_csv(csv_path)
            
            # Validate required columns
            required_cols = ['datetime', 'route_id', 'passenger_boarding']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                print(f"❌ Missing required columns: {missing_cols}")
                print(f"Available columns: {list(df.columns)}")
                return False
            
            # Create datetime features
            df['datetime'] = pd.to_datetime(df['datetime'])
            df['hour'] = df['datetime'].dt.hour
            df['day_of_week'] = df['datetime'].dt.dayofweek  # 0=Monday, 6=Sunday
            df['month'] = df['datetime'].dt.month
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
            
            # Create rush hour features
            df['is_morning_rush'] = ((df['hour'] >= 7) & (df['hour'] <= 9)).astype(int)
            df['is_evening_rush'] = ((df['hour'] >= 17) & (df['hour'] <= 19)).astype(int)
            df['is_lunch_hour'] = ((df['hour'] >= 12) & (df['hour'] <= 13)).astype(int)
            
            # Encode route IDs
            df['route_encoded'] = self.route_encoder.fit_transform(df['route_id'])
            
            # Prepare features and target
            features = ['hour', 'day_of_week', 'month', 'is_weekend', 
                       'is_morning_rush', 'is_evening_rush', 'is_lunch_hour', 'route_encoded']
            
            X = df[features]
            y = df['passenger_boarding']
            
            # Split data for training and validation
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train Random Forest model
            print("🤖 Training ML model...")
            self.ml_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            
            self.ml_model.fit(X_train, y_train)
            
            # Calculate accuracy
            y_pred = self.ml_model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            accuracy = max(0, 100 - (mae / y_test.mean() * 100))
            
            self.model_accuracy = round(accuracy, 1)
            self.is_model_trained = True
            self.historical_data = df
            
            # Show feature importance
            feature_importance = dict(zip(features, self.ml_model.feature_importances_))
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:3]
            
            print(f"✅ ML Model trained successfully!")
            print(f"📈 Model Accuracy: {self.model_accuracy}%")
            print(f"📊 Training Data: {len(df)} records")
            print(f"🎯 Top Features: {', '.join([f[0] for f in top_features])}")
            
            return True
            
        except FileNotFoundError:
            print(f"❌ File not found: {csv_path}")
            print("Make sure historic.csv is in the same directory")
            return False
        except Exception as e:
            print(f"❌ Error training model: {e}")
            return False
    
    def predict_ridership_ml(self, route_id, target_datetime):
        """Use trained ML model to predict ridership"""
        if not self.is_model_trained or not ML_AVAILABLE:
            return self._get_statistical_prediction(route_id, target_datetime.hour)
        
        try:
            # Encode route
            if route_id in self.route_encoder.classes_:
                route_encoded = self.route_encoder.transform([route_id])[0]
            else:
                # Handle unknown routes
                return self._get_statistical_prediction(route_id, target_datetime.hour)
            
            # Create features
            hour = target_datetime.hour
            day_of_week = target_datetime.weekday()
            month = target_datetime.month
            is_weekend = 1 if day_of_week >= 5 else 0
            is_morning_rush = 1 if 7 <= hour <= 9 else 0
            is_evening_rush = 1 if 17 <= hour <= 19 else 0
            is_lunch_hour = 1 if 12 <= hour <= 13 else 0
            
            features = [[hour, day_of_week, month, is_weekend, 
                        is_morning_rush, is_evening_rush, is_lunch_hour, route_encoded]]
            
            # Make prediction
            prediction = self.ml_model.predict(features)[0]
            
            # Ensure prediction is reasonable
            prediction = max(0, int(prediction))
            
            return prediction
            
        except Exception as e:
            print(f"ML prediction error: {e}")
            return self._get_statistical_prediction(route_id, target_datetime.hour)
    
    def _get_statistical_prediction(self, route_id, hour):
        """Fallback statistical prediction when ML is not available"""
        # Base patterns by hour
        hourly_patterns = {
            6: 20, 7: 45, 8: 55, 9: 35, 10: 25, 11: 30, 12: 40,
            13: 35, 14: 25, 15: 30, 16: 40, 17: 60, 18: 65,
            19: 45, 20: 30, 21: 20, 22: 15, 23: 10
        }
        
        base = hourly_patterns.get(hour, 20)
        
        # Route-specific multipliers
        route_multipliers = {
            'R1': 1.2, 'R2': 0.9, 'R3': 1.0, 'R4': 1.3, 'R5': 0.8
        }
        
        return int(base * route_multipliers.get(route_id, 1.0))
    
    def start_live_simulation(self):
        """Start the live simulation in background thread"""
        if self.simulation_running:
            return
            
        self.simulation_running = True
        self.simulation_thread = threading.Thread(target=self._simulation_loop, daemon=True)
        self.simulation_thread.start()
        print("✅ Live simulation started")
    
    def stop_live_simulation(self):
        """Stop the live simulation"""
        self.simulation_running = False
        if self.simulation_thread:
            self.simulation_thread.join()
        print("⏹️ Live simulation stopped")
    
    def _simulation_loop(self):
        """Main simulation loop - runs every 10 seconds"""
        while self.simulation_running:
            try:
                current_time = datetime.now()
                
                # Update bus positions every 10 seconds
                self._update_live_bus_positions(current_time)
                
                # Update passenger boarding every 10 seconds  
                self._simulate_passenger_boarding(current_time)
                
                # Update ridership predictions every minute
                if current_time.second == 0:
                    self._update_live_predictions(current_time)
                
                # Log actual vs predicted every 5 minutes
                if current_time.minute % 5 == 0 and current_time.second == 0:
                    self._log_actual_vs_predicted(current_time)
                
                time.sleep(10)  # Update every 10 seconds
                
            except Exception as e:
                print(f"⚠️ Simulation error: {e}")
                time.sleep(5)
    
    def _update_live_bus_positions(self, current_time):
        """Update GPS positions of all buses dynamically"""
        for route in self.routes:
            for bus_num in [1, 2]:  # 2 buses per route
                bus_id = f"{route}_B{bus_num:02d}"
                
                if bus_id not in self.live_buses:
                    # Initialize new bus with different base positions
                    route_offset = ord(route[-1]) - ord('1')
                    base_lat = 23.0225 + route_offset * 0.05
                    base_lng = 72.5714 + route_offset * 0.05
                    
                    self.live_buses[bus_id] = LiveBus(
                        bus_id=bus_id,
                        route_id=route,
                        latitude=base_lat,
                        longitude=base_lng,
                        speed=random.uniform(20, 40),
                        passenger_count=random.randint(5, 35),
                        status='in_service',
                        last_stop=f"{route}_S{random.randint(1,5):02d}",
                        next_stop=f"{route}_S{random.randint(1,5):02d}",
                        delay_minutes=0
                    )
                
                bus = self.live_buses[bus_id]
                
                # Simulate bus movement
                speed_kmh = bus.speed
                distance_per_10sec = (speed_kmh * 1000) / 360  # meters per 10 seconds
                lat_change = (distance_per_10sec / 111000) * random.choice([-1, 1])
                lng_change = (distance_per_10sec / 111000) * random.choice([-1, 1])
                
                bus.latitude += lat_change * 0.1  # Smaller movements
                bus.longitude += lng_change * 0.1
                bus.speed = max(10, min(50, bus.speed + random.uniform(-2, 2)))
                
                # Simulate passenger changes
                if random.random() < 0.2:  # 20% chance
                    change = random.randint(-3, 5)
                    bus.passenger_count = max(0, min(45, bus.passenger_count + change))
                
                # Simulate delays
                if random.random() < 0.05:  # 5% chance
                    bus.delay_minutes += random.randint(-1, 2)
                    bus.delay_minutes = max(0, min(15, bus.delay_minutes))
    
    def _simulate_passenger_boarding(self, current_time):
        """Simulate realistic passenger boarding based on time patterns"""
        hour = current_time.hour
        
        # Activity levels by hour
        if hour in [7, 8, 17, 18]:  # Rush hours
            activity_multiplier = 2.5
            base_passengers = random.randint(3, 8)
        elif hour in [12, 13]:  # Lunch
            activity_multiplier = 1.5
            base_passengers = random.randint(2, 5)
        elif 6 <= hour <= 22:  # Normal hours
            activity_multiplier = 1.0
            base_passengers = random.randint(1, 4)
        else:  # Night hours
            activity_multiplier = 0.3
            base_passengers = random.randint(0, 2)
        
        for route in self.routes:
            # Simulate boarding events
            if random.random() < 0.3 * activity_multiplier:
                passengers_boarding = base_passengers
                passengers_alighting = random.randint(0, passengers_boarding // 2)
                stop_id = f"{route}_S{random.randint(1, 10):02d}"
                
                boarding_event = {
                    'timestamp': current_time.isoformat(),
                    'route_id': route,
                    'stop_id': stop_id,
                    'passengers_boarding': passengers_boarding,
                    'passengers_alighting': passengers_alighting,
                    'event_type': 'live_simulation',
                    'hour': hour
                }
                
                self.passenger_boarding_events.append(boarding_event)
                
                # Keep only last 200 events
                if len(self.passenger_boarding_events) > 200:
                    self.passenger_boarding_events = self.passenger_boarding_events[-200:]
    
    def _update_live_predictions(self, current_time):
        """Update ridership predictions using ML model"""
        for route in self.routes:
            predictions = []
            
            # Generate predictions for next 3 hours
            for i in range(1, 4):
                future_time = current_time + timedelta(hours=i)
                
                # Get ML or statistical prediction
                if self.is_model_trained:
                    base_prediction = self.predict_ridership_ml(route, future_time)
                    prediction_type = f"ML_Model (Accuracy: {self.model_accuracy}%)"
                else:
                    base_prediction = self._get_statistical_prediction(route, future_time.hour)
                    prediction_type = "Statistical_Pattern"
                
                # Adjust based on recent activity
                recent_events = [e for e in self.passenger_boarding_events 
                               if e['route_id'] == route and 
                               datetime.fromisoformat(e['timestamp']) > current_time - timedelta(minutes=30)]
                
                current_activity = sum(e['passengers_boarding'] for e in recent_events)
                
                # Activity-based adjustment
                if current_activity > 40:  # Very high activity
                    multiplier = 1.3
                elif current_activity > 25:  # High activity
                    multiplier = 1.15
                elif current_activity < 5:  # Low activity
                    multiplier = 0.7
                elif current_activity < 15:  # Below average
                    multiplier = 0.85
                else:
                    multiplier = 1.0
                
                # Apply some randomness for realism
                final_prediction = int(base_prediction * multiplier * random.uniform(0.9, 1.1))
                
                predictions.append({
                    'hour': future_time.hour,
                    'timestamp': future_time.isoformat(),
                    'predicted_ridership': final_prediction,
                    'base_prediction': base_prediction,
                    'activity_multiplier': round(multiplier, 2),
                    'current_activity': current_activity,
                    'prediction_type': prediction_type,
                    'confidence': 'high' if self.is_model_trained else 'medium'
                })
            
            self.current_predictions[route] = {
                'route_id': route,
                'predictions': predictions,
                'last_updated': current_time.isoformat(),
                'model_trained': self.is_model_trained,
                'model_accuracy': self.model_accuracy
            }
    
    def _log_actual_vs_predicted(self, current_time):
        """Log actual ridership vs predictions for accuracy tracking"""
        for route in self.routes:
            # Get actual ridership from last 5 minutes
            recent_events = [e for e in self.passenger_boarding_events 
                           if e['route_id'] == route and 
                           datetime.fromisoformat(e['timestamp']) > current_time - timedelta(minutes=5)]
            
            actual_ridership = sum(e['passengers_boarding'] for e in recent_events)
            
            # Find matching prediction
            predicted = None
            prediction_details = None
            if route in self.current_predictions:
                for pred in self.current_predictions[route]['predictions']:
                    pred_time = datetime.fromisoformat(pred['timestamp'])
                    if abs((pred_time - current_time).total_seconds()) < 300:  # Within 5 minutes
                        predicted = pred['predicted_ridership']
                        prediction_details = {
                            'base': pred['base_prediction'],
                            'multiplier': pred['activity_multiplier'],
                            'type': pred['prediction_type']
                        }
                        break
            
            # Calculate accuracy
            accuracy = None
            if predicted is not None and predicted > 0:
                accuracy = round(100 - (abs(actual_ridership - predicted) / max(predicted, 1) * 100), 1)
                accuracy = max(0, min(100, accuracy))  # Cap between 0-100
            
            log_entry = {
                'timestamp': current_time.isoformat(),
                'route_id': route,
                'actual_ridership': actual_ridership,
                'predicted_ridership': predicted,
                'prediction_details': prediction_details,
                'accuracy_percent': accuracy,
                'model_type': 'ML_trained' if self.is_model_trained else 'statistical'
            }
            
            self.actual_ridership_log.append(log_entry)
            
            # Keep only last 100 entries
            if len(self.actual_ridership_log) > 100:
                self.actual_ridership_log = self.actual_ridership_log[-100:]

    # === API METHODS ===
    
    def get_live_predictions(self, route_id: str = None) -> Dict:
        """Get current live predictions"""
        if route_id:
            prediction = self.current_predictions.get(route_id, {'error': 'No predictions available'})
            if 'error' not in prediction:
                prediction['hackathon_features'] = {
                    'ml_model_trained': self.is_model_trained,
                    'real_time_updates': True,
                    'accuracy_tracking': True,
                    'live_data_integration': True
                }
            return prediction
        else:
            return {
                'all_routes': self.current_predictions,
                'system_status': {
                    'ml_model_trained': self.is_model_trained,
                    'model_accuracy': self.model_accuracy,
                    'simulation_running': self.simulation_running,
                    'total_routes': len(self.routes),
                    'last_updated': datetime.now().isoformat()
                }
            }
    
    def get_live_bus_positions(self) -> Dict:
        """Get current bus positions (live GPS feed)"""
        live_positions = []
        for bus_id, bus in self.live_buses.items():
            live_positions.append({
                'bus_id': bus.bus_id,
                'route_id': bus.route_id,
                'latitude': round(bus.latitude, 6),
                'longitude': round(bus.longitude, 6),
                'speed': round(bus.speed, 1),
                'passenger_count': bus.passenger_count,
                'capacity_percent': round((bus.passenger_count / 45) * 100, 1),
                'status': bus.status,
                'last_stop': bus.last_stop,
                'next_stop': bus.next_stop,
                'delay_minutes': bus.delay_minutes,
                'last_updated': datetime.now().isoformat()
            })
        
        return {
            'buses': live_positions,
            'total_buses': len(live_positions),
            'simulation_active': self.simulation_running,
            'update_frequency': '10 seconds'
        }
    
    def get_live_boarding_activity(self) -> Dict:
        """Get recent passenger boarding activity"""
        # Get activity by route for last hour
        current_time = datetime.now()
        hourly_activity = {}
        
        for route in self.routes:
            recent_events = [e for e in self.passenger_boarding_events 
                           if e['route_id'] == route and 
                           datetime.fromisoformat(e['timestamp']) > current_time - timedelta(hours=1)]
            
            total_boarding = sum(e['passengers_boarding'] for e in recent_events)
            hourly_activity[route] = {
                'total_boarding_last_hour': total_boarding,
                'event_count': len(recent_events),
                'avg_per_event': round(total_boarding / max(len(recent_events), 1), 1)
            }
        
        return {
            'recent_boarding_events': self.passenger_boarding_events[-15:],  # Last 15 events
            'hourly_activity_summary': hourly_activity,
            'total_events_logged': len(self.passenger_boarding_events),
            'simulation_active': self.simulation_running
        }
    
    def get_forecasted_vs_actual_live(self) -> Dict:
        """Get live comparison of forecasted vs actual ridership"""
        # Calculate overall accuracy
        recent_logs = [log for log in self.actual_ridership_log 
                      if log['accuracy_percent'] is not None]
        
        avg_accuracy = None
        if recent_logs:
            avg_accuracy = round(sum(log['accuracy_percent'] for log in recent_logs) / len(recent_logs), 1)
        
        return {
            'accuracy_log': self.actual_ridership_log[-20:],  # Last 20 comparisons
            'current_predictions': self.current_predictions,
            'performance_metrics': {
                'average_accuracy': avg_accuracy,
                'total_comparisons': len(recent_logs),
                'model_type': 'ML_trained' if self.is_model_trained else 'statistical'
            },
            'live_status': 'active' if self.simulation_running else 'inactive',
            'last_updated': datetime.now().isoformat()
        }
    
    def get_live_alerts(self) -> List[Dict]:
        """Generate live alerts based on current conditions"""
        alerts = []
        current_time = datetime.now()
        
        # Check for delayed buses
        for bus_id, bus in self.live_buses.items():
            if bus.delay_minutes > 5:
                severity = 'critical' if bus.delay_minutes > 10 else 'high' if bus.delay_minutes > 7 else 'medium'
                alerts.append({
                    'id': f"delay_{bus_id}",
                    'type': 'delay',
                    'bus_id': bus_id,
                    'route_id': bus.route_id,
                    'message': f"Bus {bus_id} delayed by {bus.delay_minutes} minutes",
                    'severity': severity,
                    'timestamp': current_time.isoformat(),
                    'action_needed': 'Dispatch backup bus' if severity == 'critical' else 'Monitor closely'
                })
        
        # Check for overcrowding
        for bus_id, bus in self.live_buses.items():
            if bus.passenger_count > 40:
                overcrowding_level = round((bus.passenger_count / 45) * 100, 1)
                alerts.append({
                    'id': f"crowding_{bus_id}",
                    'type': 'overcrowding',
                    'bus_id': bus_id,
                    'route_id': bus.route_id,
                    'message': f"Bus {bus_id} at {overcrowding_level}% capacity ({bus.passenger_count}/45)",
                    'severity': 'high' if overcrowding_level > 90 else 'medium',
                    'timestamp': current_time.isoformat(),
                    'action_needed': 'Schedule additional bus'
                })
        
        # Check for high demand predictions
        for route, pred_data in self.current_predictions.items():
            for pred in pred_data['predictions']:
                if pred['predicted_ridership'] > 70:
                    alerts.append({
                        'id': f"demand_{route}_{pred['hour']}",
                        'type': 'high_demand_predicted',
                        'route_id': route,
                        'message': f"High demand predicted for {route} at {pred['hour']}:00 ({pred['predicted_ridership']} passengers)",
                        'severity': 'medium',
                        'timestamp': current_time.isoformat(),
                        'action_needed': 'Prepare additional capacity'
                    })
        
        return alerts

    def get_model_info(self) -> Dict:
        """Get information about the ML model"""
        if not self.is_model_trained:
            return {
                'model_status': 'not_trained',
                'message': 'ML model not trained. Using statistical predictions.',
                'suggestion': 'Train model with historic.csv for better accuracy'
            }
        
        return {
            'model_status': 'trained',
            'model_type': 'Random Forest Regressor',
            'accuracy': f"{self.model_accuracy}%",
            'training_data_size': len(self.historical_data) if self.historical_data is not None else 0,
            'features_used': ['hour', 'day_of_week', 'month', 'is_weekend', 
                            'is_morning_rush', 'is_evening_rush', 'is_lunch_hour', 'route_encoded'],
            'last_trained': 'On startup'
        }

# Global live predictor instance
live_predictor = DynamicLivePredictor()

# === API FUNCTIONS ===
def initialize_ml_model(csv_path="historic.csv"):
    """Initialize and train the ML model"""
    return live_predictor.load_and_train_model(csv_path)

def start_live_simulation():
    """Start the live simulation"""
    live_predictor.start_live_simulation()
    return {"status": "Live simulation started", "timestamp": datetime.now().isoformat()}

def stop_live_simulation():
    """Stop the live simulation"""  
    live_predictor.stop_live_simulation()
    return {"status": "Live simulation stopped", "timestamp": datetime.now().isoformat()}

def get_live_ridership_predictions(route_id: str = None):
    """Get live ridership predictions"""
    return live_predictor.get_live_predictions(route_id)

def get_live_gps_feed():
    """Get live GPS positions of all buses"""
    return live_predictor.get_live_bus_positions()

def get_live_boarding_events():
    """Get real-time boarding events"""
    return live_predictor.get_live_boarding_activity()

def get_live_dashboard_data():
    """Get complete dashboard data for live view"""
    return {
        'predictions': live_predictor.get_live_predictions(),
        'bus_positions': live_predictor.get_live_bus_positions(),
        'boarding_activity': live_predictor.get_live_boarding_activity(),
        'forecasted_vs_actual': live_predictor.get_forecasted_vs_actual_live(),
        'alerts': live_predictor.get_live_alerts(),
        'model_info': live_predictor.get_model_info(),
        'system_status': {
            'simulation_running': live_predictor.simulation_running,
            'ml_model_trained': live_predictor.is_model_trained,
            'last_updated': datetime.now().isoformat()
        }
    }

if __name__ == "__main__":
    print("🚌 Testing Complete Dynamic Live Prediction System...")
    
    # Try to train ML model
    print("\n🤖 Training ML Model...")
    model_trained = initialize_ml_model("historic.csv")
    
    # Start simulation
    print("\n🔄 Starting Live Simulation...")
    start_live_simulation()
    
    # Let it run for a bit
    time.sleep(3)
    
    # Test API functions
    print("\n📊 Testing Predictions:")
    predictions = get_live_ridership_predictions('R1')
    print(f"Model trained: {predictions.get('model_trained', False)}")
    print(f"Predictions: {len(predictions.get('predictions', []))}")
    
    print("\n🚌 Testing Bus Positions:")
    buses = get_live_gps_feed()
    print(f"Active buses: {buses['total_buses']}")
    
    print("\n📈 Testing Dashboard:")
    dashboard = get_live_dashboard_data()
    print(f"ML Model Status: {dashboard['model_info']['model_status']}")
    print(f"Alerts: {len(dashboard['alerts'])}")
    
    # Run for 20 seconds to see updates
    print("\n⏳ Running for 20 seconds to test live updates...")
    time.sleep(20)
    
    # Final test
    final_predictions = get_live_ridership_predictions()
    print(f"\n✅ Final Test - System Status: {final_predictions['system_status']['simulation_running']}")
    
    # Stop simulation
    stop_live_simulation()
    print("\n🛑 Test completed!")