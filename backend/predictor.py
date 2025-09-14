# dynamic_predictor.py - Real-time Dynamic Simulation
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import json
import random
import asyncio
import threading
import time
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

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
        self.models = {}
        self.live_buses = {}
        self.live_ridership = {}
        self.historical_data = {}
        self.simulation_running = False
        self.simulation_thread = None
        self.routes = ['R1', 'R2', 'R3', 'R4', 'R5']
        
        # Real-time state
        self.current_predictions = {}
        self.actual_ridership_log = []
        self.gps_positions = {}
        self.passenger_boarding_events = []
        
        print("🚀 Dynamic Live Predictor initialized")
    
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
                    # Initialize new bus
                    base_lat = 23.0225 + (ord(route[-1]) - ord('1')) * 0.05
                    base_lng = 72.5714 + (ord(route[-1]) - ord('1')) * 0.05
                    
                    self.live_buses[bus_id] = LiveBus(
                        bus_id=bus_id,
                        route_id=route,
                        latitude=base_lat,
                        longitude=base_lng,
                        speed=random.uniform(20, 40),
                        passenger_count=random.randint(5, 35),
                        status='in_service',
                        last_stop=f"{route}_S01",
                        next_stop=f"{route}_S02",
                        delay_minutes=0
                    )
                
                bus = self.live_buses[bus_id]
                
                # Move bus along route (simulate movement)
                speed_kmh = bus.speed
                distance_per_10sec = (speed_kmh * 1000) / 360  # meters per 10 seconds
                lat_change = (distance_per_10sec / 111000) * random.choice([-1, 1])
                lng_change = (distance_per_10sec / 111000) * random.choice([-1, 1])
                
                bus.latitude += lat_change
                bus.longitude += lng_change
                bus.speed = max(10, min(50, bus.speed + random.uniform(-3, 3)))
                
                # Simulate passenger changes
                if random.random() < 0.3:  # 30% chance of passenger change
                    change = random.randint(-5, 8)
                    bus.passenger_count = max(0, min(45, bus.passenger_count + change))
                
                # Simulate delays
                if random.random() < 0.1:  # 10% chance of delay change
                    bus.delay_minutes += random.randint(-2, 3)
                    bus.delay_minutes = max(0, bus.delay_minutes)
    
    def _simulate_passenger_boarding(self, current_time):
        """Simulate passengers boarding/alighting in real-time"""
        hour = current_time.hour
        minute = current_time.minute
        
        # Peak hours have more activity
        if hour in [7, 8, 17, 18]:  # Rush hours
            activity_multiplier = 2.0
        elif hour in [12, 13]:  # Lunch
            activity_multiplier = 1.3
        else:
            activity_multiplier = 1.0
        
        for route in self.routes:
            # Simulate boarding events
            if random.random() < 0.4 * activity_multiplier:  # Boarding event
                passengers = random.randint(1, 6)
                stop_id = f"{route}_S{random.randint(1, 10):02d}"
                
                boarding_event = {
                    'timestamp': current_time.isoformat(),
                    'route_id': route,
                    'stop_id': stop_id,
                    'passengers_boarding': passengers,
                    'passengers_alighting': random.randint(0, 3),
                    'event_type': 'live'
                }
                
                self.passenger_boarding_events.append(boarding_event)
                
                # Keep only last 100 events
                if len(self.passenger_boarding_events) > 100:
                    self.passenger_boarding_events = self.passenger_boarding_events[-100:]
    
    def _update_live_predictions(self, current_time):
        """Update ridership predictions based on current patterns"""
        for route in self.routes:
            # Get recent boarding activity
            recent_events = [e for e in self.passenger_boarding_events 
                           if e['route_id'] == route and 
                           datetime.fromisoformat(e['timestamp']) > current_time - timedelta(minutes=30)]
            
            current_activity = sum(e['passengers_boarding'] for e in recent_events)
            
            # Generate prediction for next 3 hours
            predictions = []
            for i in range(1, 4):
                future_time = current_time + timedelta(hours=i)
                
                # Base prediction
                base = self._get_base_prediction(route, future_time.hour)
                
                # Adjust based on current activity
                if current_activity > 30:  # High activity
                    multiplier = 1.2
                elif current_activity < 10:  # Low activity  
                    multiplier = 0.8
                else:
                    multiplier = 1.0
                
                prediction = int(base * multiplier * random.uniform(0.9, 1.1))
                
                predictions.append({
                    'hour': future_time.hour,
                    'timestamp': future_time.isoformat(),
                    'predicted_ridership': prediction,
                    'confidence': 'live_updated',
                    'based_on_current_activity': current_activity
                })
            
            self.current_predictions[route] = {
                'route_id': route,
                'predictions': predictions,
                'last_updated': current_time.isoformat()
            }
    
    def _get_base_prediction(self, route, hour):
        """Get base prediction for route at given hour"""
        patterns = {
            7: 45, 8: 55, 9: 35, 10: 25, 11: 30, 12: 40,
            13: 35, 14: 25, 15: 30, 16: 40, 17: 60, 18: 65,
            19: 45, 20: 30, 21: 20, 22: 15
        }
        
        base = patterns.get(hour, 20)
        
        # Route-specific multipliers
        route_multipliers = {'R1': 1.2, 'R2': 0.9, 'R3': 1.0, 'R4': 1.3, 'R5': 0.8}
        return base * route_multipliers.get(route, 1.0)
    
    def _log_actual_vs_predicted(self, current_time):
        """Log actual ridership vs what was predicted"""
        for route in self.routes:
            # Get actual ridership from recent boarding events
            recent_events = [e for e in self.passenger_boarding_events 
                           if e['route_id'] == route and 
                           datetime.fromisoformat(e['timestamp']) > current_time - timedelta(minutes=5)]
            
            actual_ridership = sum(e['passengers_boarding'] for e in recent_events)
            
            # Get what was predicted for this time (if available)
            predicted = None
            if route in self.current_predictions:
                # Find closest prediction
                for pred in self.current_predictions[route]['predictions']:
                    pred_time = datetime.fromisoformat(pred['timestamp'])
                    if abs((pred_time - current_time).total_seconds()) < 300:  # Within 5 minutes
                        predicted = pred['predicted_ridership']
                        break
            
            log_entry = {
                'timestamp': current_time.isoformat(),
                'route_id': route,
                'actual_ridership': actual_ridership,
                'predicted_ridership': predicted,
                'accuracy': None if predicted is None else round(100 - abs(actual_ridership - predicted) / max(predicted, 1) * 100, 1)
            }
            
            self.actual_ridership_log.append(log_entry)
            
            # Keep only last 50 entries
            if len(self.actual_ridership_log) > 50:
                self.actual_ridership_log = self.actual_ridership_log[-50:]

    # === API METHODS ===
    
    def get_live_predictions(self, route_id: str = None) -> Dict:
        """Get current live predictions"""
        if route_id:
            return self.current_predictions.get(route_id, {'error': 'No predictions available'})
        else:
            return {
                'all_routes': self.current_predictions,
                'last_updated': datetime.now().isoformat(),
                'simulation_status': 'running' if self.simulation_running else 'stopped'
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
                'status': bus.status,
                'last_stop': bus.last_stop,
                'next_stop': bus.next_stop,
                'delay_minutes': bus.delay_minutes,
                'last_updated': datetime.now().isoformat()
            })
        
        return {
            'buses': live_positions,
            'total_buses': len(live_positions),
            'simulation_active': self.simulation_running
        }
    
    def get_live_boarding_activity(self) -> Dict:
        """Get recent passenger boarding activity"""
        return {
            'recent_boarding_events': self.passenger_boarding_events[-10:],  # Last 10 events
            'total_events_logged': len(self.passenger_boarding_events),
            'simulation_active': self.simulation_running
        }
    
    def get_forecasted_vs_actual_live(self) -> Dict:
        """Get live comparison of forecasted vs actual ridership"""
        return {
            'accuracy_log': self.actual_ridership_log[-10:],  # Last 10 comparisons
            'current_predictions': self.current_predictions,
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
                alerts.append({
                    'id': f"delay_{bus_id}",
                    'type': 'delay',
                    'bus_id': bus_id,
                    'route_id': bus.route_id,
                    'message': f"Bus {bus_id} delayed by {bus.delay_minutes} minutes",
                    'severity': 'high' if bus.delay_minutes > 10 else 'medium',
                    'timestamp': current_time.isoformat()
                })
        
        # Check for overcrowding
        for bus_id, bus in self.live_buses.items():
            if bus.passenger_count > 40:
                alerts.append({
                    'id': f"crowding_{bus_id}",
                    'type': 'overcrowding',
                    'bus_id': bus_id,
                    'route_id': bus.route_id,
                    'message': f"Bus {bus_id} overcrowded ({bus.passenger_count}/45 capacity)",
                    'severity': 'medium',
                    'timestamp': current_time.isoformat()
                })
        
        return alerts

# Global live predictor instance
live_predictor = DynamicLivePredictor()

# === API FUNCTIONS ===
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
        'system_status': {
            'simulation_running': live_predictor.simulation_running,
            'last_updated': datetime.now().isoformat()
        }
    }

if __name__ == "__main__":
    print("🚌 Testing Dynamic Live Prediction System...")
    
    # Start simulation
    start_live_simulation()
    
    # Let it run for a bit
    time.sleep(5)
    
    # Test API functions
    print("\n📊 Live Predictions:")
    predictions = get_live_ridership_predictions('R1')
    print(json.dumps(predictions, indent=2))
    
    print("\n🚌 Live Bus Positions:")
    buses = get_live_gps_feed()
    print(f"Found {buses['total_buses']} buses")
    
    print("\n📈 Dashboard Data:")
    dashboard = get_live_dashboard_data()
    print(f"Alerts: {len(dashboard['alerts'])}")
    print(f"Simulation Status: {dashboard['system_status']['simulation_running']}")
    
    # Let it run for 30 seconds to see updates
    print("\n⏳ Running simulation for 30 seconds...")
    time.sleep(30)
    
    # Stop simulation
    stop_live_simulation()
    print("\n✅ Test completed!")