import pandas as pd
from typing import List, Dict, Optional
from datetime import datetime
import os
import random

# Import models from your models.py file
from models import BusStop, GPSSimulator
class BusDataManager:
    def __init__(self, data_folder="../data"):
        self.data_folder = data_folder
        self.routes_df = None
        self.stops_df = None
        self.passenger_df = None
        self.gps_simulators = {}  # Store GPS simulators for each route
        self.load_all_data()
    
    def load_all_data(self):
        """Load all CSV files"""
        try:
            # Load routes data
            routes_path = os.path.join(self.data_folder, "routes.csv")
            self.routes_df = pd.read_csv(routes_path)
            print(f"✅ Loaded {len(self.routes_df)} routes")
            
            # Load stops data
            stops_path = os.path.join(self.data_folder, "stops.csv")
            self.stops_df = pd.read_csv(stops_path)
            print(f"✅ Loaded {len(self.stops_df)} stops")
            
            # Load passenger data
            passenger_path = os.path.join(self.data_folder, "passenger_data.csv")
            self.passenger_df = pd.read_csv(passenger_path)
            self.passenger_df['datetime'] = pd.to_datetime(
                self.passenger_df['date'] + ' ' + self.passenger_df['time']
            )
            print(f"✅ Loaded {len(self.passenger_df)} passenger records")
            
            # Initialize GPS simulators for each route
            self._initialize_gps_simulators()
            
        except FileNotFoundError as e:
            print(f"❌ Error loading files: {e}")
            print(f"Expected files in: {os.path.abspath(self.data_folder)}")
            # Create sample data if files don't exist
            self._create_sample_data()
    
    def _initialize_gps_simulators(self):
        """Initialize GPS simulators for each route"""
        for route_id in self.routes_df['route_id'].unique():
            route_stops = self.get_stops_for_route(route_id)
            if route_stops:
                bus_stops = [
                    BusStop(**stop) for stop in route_stops
                ]
                self.gps_simulators[route_id] = GPSSimulator(bus_stops)
    
    def _create_sample_data(self):
        """Create minimal sample data if CSV files don't exist"""
        print("📝 Creating sample data...")
        
        # Sample routes
        sample_routes = pd.DataFrame([
            {
                'route_id': 1,
                'route_name': 'Test Route 1',
                'city': 'Mumbai',
                'start_stop': 'Start Point',
                'end_stop': 'End Point',
                'total_stops': 5,
                'estimated_time_minutes': 30,
                'frequency_minutes': 10
            }
        ])
        
        # Sample stops
        sample_stops = pd.DataFrame([
            {'stop_id': 1, 'stop_name': 'Stop 1', 'route_id': 1, 'latitude': 19.0760, 'longitude': 72.8777, 'stop_sequence': 1, 'city': 'Mumbai'},
            {'stop_id': 2, 'stop_name': 'Stop 2', 'route_id': 1, 'latitude': 19.0860, 'longitude': 72.8777, 'stop_sequence': 2, 'city': 'Mumbai'},
        ])
        
        # Sample passenger data
        sample_passenger = pd.DataFrame([
            {
                'date': '2024-01-15',
                'time': '08:30',
                'route_id': 1,
                'stop_id': 1,
                'boarding': 25,
                'alighting': 5,
                'current_occupancy': 45,
                'day_type': 'weekday',
                'weather': 'clear'
            }
        ])
        
        self.routes_df = sample_routes
        self.stops_df = sample_stops
        self.passenger_df = sample_passenger
        self.passenger_df['datetime'] = pd.to_datetime(self.passenger_df['date'] + ' ' + self.passenger_df['time'])
        
        print("✅ Sample data created")
    
    def get_all_routes(self) -> List[Dict]:
        """Get all routes"""
        return self.routes_df.to_dict('records')
    
    def get_routes_by_city(self, city: str) -> List[Dict]:
        """Get routes for a specific city"""
        city_routes = self.routes_df[self.routes_df['city'].str.lower() == city.lower()]
        return city_routes.to_dict('records')
    
    def get_stops_for_route(self, route_id: int) -> List[Dict]:
        """Get stops for a route"""
        route_stops = self.stops_df[self.stops_df['route_id'] == route_id].sort_values('stop_sequence')
        return route_stops.to_dict('records')
    
    def get_passenger_data_for_route(self, route_id: int, date_filter: Optional[str] = None) -> List[Dict]:
        """Get passenger data for a route"""
        route_data = self.passenger_df[self.passenger_df['route_id'] == route_id]
        
        if date_filter:
            route_data = route_data[route_data['date'] == date_filter]
        
        return route_data.to_dict('records')
    
    def get_peak_hours_analysis(self, route_id: int) -> Dict:
        """Analyze peak hours for a route"""
        route_data = self.passenger_df[self.passenger_df['route_id'] == route_id]
        route_data['hour'] = pd.to_datetime(route_data['time'], format='%H:%M').dt.hour
        
        peak_analysis = route_data.groupby('hour').agg({
            'boarding': 'sum',
            'alighting': 'sum',
            'current_occupancy': 'mean'
        }).round(2)
        
        return {
            'boarding': peak_analysis['boarding'].to_dict(),
            'alighting': peak_analysis['alighting'].to_dict(),
            'occupancy': peak_analysis['current_occupancy'].to_dict()
        }
    
    def get_optimization_comparison(self, route_id: int) -> Dict:
        """Get before vs after optimization comparison"""
        # Mock optimization results for demo
        before_stats = {
            "avg_wait_time": 12.5,
            "bus_utilization": 45.2,
            "bunching_incidents": 15,
            "on_time_performance": 62.3
        }
        
        after_stats = {
            "avg_wait_time": 7.8,
            "bus_utilization": 68.7,
            "bunching_incidents": 4,
            "on_time_performance": 84.1
        }
        
        improvements = {
            "wait_time_reduction": round(((before_stats["avg_wait_time"] - after_stats["avg_wait_time"]) / before_stats["avg_wait_time"]) * 100, 1),
            "utilization_increase": round(((after_stats["bus_utilization"] - before_stats["bus_utilization"]) / before_stats["bus_utilization"]) * 100, 1),
            "bunching_reduction": round(((before_stats["bunching_incidents"] - after_stats["bunching_incidents"]) / before_stats["bunching_incidents"]) * 100, 1),
            "performance_improvement": round(after_stats["on_time_performance"] - before_stats["on_time_performance"], 1)
        }
        
        return {
            "before": before_stats,
            "after": after_stats,
            "improvements": improvements
        }
    
    def simulate_live_buses(self, route_id: int) -> List[Dict]:
        """Simulate live bus positions"""
        import random
        
        stops = self.get_stops_for_route(route_id)
        if not stops:
            return []
        
        live_buses = []
        
        # Simulate 3 buses on the route
        for bus_num in range(1, 4):
            random_stop_index = random.randint(0, len(stops)-2)
            current_stop = stops[random_stop_index]
            next_stop = stops[random_stop_index + 1] if random_stop_index + 1 < len(stops) else stops[random_stop_index]
            
            # Calculate position between current and next stop
            lat_diff = next_stop['latitude'] - current_stop['latitude']
            lng_diff = next_stop['longitude'] - current_stop['longitude']
            progress = random.random()
            
            bus_data = {
                "bus_id": f"BUS_{route_id}_{bus_num:02d}",
                "route_id": route_id,
                "current_lat": current_stop['latitude'] + (lat_diff * progress),
                "current_lng": current_stop['longitude'] + (lng_diff * progress),
                "next_stop": next_stop['stop_name'],
                "next_stop_id": next_stop['stop_id'],
                "eta_minutes": random.randint(2, 8),
                "current_occupancy": random.randint(20, 120),
                "max_capacity": 150,
                "status": random.choice(["on_time", "delayed", "ahead"]),
                "delay_minutes": random.randint(-3, 10) if random.choice([True, False]) else 0,
                "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            live_buses.append(bus_data)
        
        return live_buses
    
    def get_system_summary(self) -> Dict:
        """Get system summary statistics"""
        if self.routes_df is None:
            return {"error": "No data loaded"}
        
        summary = {
            'total_routes': len(self.routes_df),
            'total_stops': len(self.stops_df),
            'cities': self.routes_df['city'].unique().tolist(),
            'date_range': {
                'start': self.passenger_df['date'].min(),
                'end': self.passenger_df['date'].max()
            } if not self.passenger_df.empty else None
        }
        
        return summary

# Global instance
data_manager = BusDataManager()