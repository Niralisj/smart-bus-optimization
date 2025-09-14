from pydantic import BaseModel, field_validator
from typing import Optional, List
from datetime import datetime
import random
import math

class BusRoute(BaseModel):
    route_id: int
    route_name: str
    city: str
    start_stop: str
    end_stop: str
    total_stops: int
    estimated_time_minutes: int
    frequency_minutes: int

class BusStop(BaseModel):
    stop_id: int
    stop_name: str
    route_id: int
    latitude: float
    longitude: float
    stop_sequence: int
    city: str

class PassengerData(BaseModel):
    date: str
    time: str
    route_id: int
    stop_id: int
    boarding: int
    alighting: int
    current_occupancy: int
    day_type: str
    weather: str

class LiveBus(BaseModel):
    bus_id: str
    route_id: int
    current_lat: float
    current_lng: float
    next_stop: str
    next_stop_id: int
    eta_minutes: int
    current_occupancy: int
    max_capacity: int
    status: str
    delay_minutes: int
    last_updated: str

class GPSCoordinate(BaseModel):
    latitude: float
    longitude: float
    
    @field_validator('latitude', mode='before')
    @classmethod
    def validate_latitude(cls, v):
        if not -90 <= float(v) <= 90:
            raise ValueError('Latitude must be between -90 and 90')
        return float(v)
    
    @field_validator('longitude', mode='before')
    @classmethod
    def validate_longitude(cls, v):
        if not -180 <= float(v) <= 180:
            raise ValueError('Longitude must be between -180 and 180')
        return float(v)

class OptimizationResult(BaseModel):
    route_id: int
    before_stats: dict
    after_stats: dict
    improvements: dict

class SystemAlert(BaseModel):
    id: int
    type: str
    route_id: int
    message: str
    severity: str
    timestamp: str
    action: str

class GPSSimulator:
    """Simple GPS simulator for live bus tracking"""
    
    def __init__(self, route_stops: List[BusStop]):
        self.route_stops = route_stops
        self.current_position = 0
        self.progress_between_stops = 0.0
        
    def get_current_position(self) -> GPSCoordinate:
        """Get current GPS position between two stops"""
        if len(self.route_stops) < 2:
            return GPSCoordinate(
                latitude=self.route_stops[0].latitude,
                longitude=self.route_stops[0].longitude
            )
        
        # Get current and next stop
        current_stop = self.route_stops[self.current_position]
        next_stop_index = min(self.current_position + 1, len(self.route_stops) - 1)
        next_stop = self.route_stops[next_stop_index]
        
        # Calculate interpolated position
        lat_diff = next_stop.latitude - current_stop.latitude
        lng_diff = next_stop.longitude - current_stop.longitude
        
        current_lat = current_stop.latitude + (lat_diff * self.progress_between_stops)
        current_lng = current_stop.longitude + (lng_diff * self.progress_between_stops)
        
        return GPSCoordinate(latitude=current_lat, longitude=current_lng)
    
    def move_forward(self, speed_factor: float = 0.1):
        """Move the bus forward along the route"""
        self.progress_between_stops += speed_factor
        
        if self.progress_between_stops >= 1.0:
            self.progress_between_stops = 0.0
            self.current_position = min(self.current_position + 1, len(self.route_stops) - 1)
    
    def get_next_stop_info(self) -> Optional[BusStop]:
        """Get information about the next stop"""
        next_index = min(self.current_position + 1, len(self.route_stops) - 1)
        if next_index < len(self.route_stops):
            return self.route_stops[next_index]
        return None
    
    def calculate_eta(self) -> int:
        """Calculate ETA to next stop in minutes"""
        # Simple calculation based on progress and random factors
        remaining_progress = 1.0 - self.progress_between_stops
        base_time = 3  # 3 minutes base time between stops
        eta = int(base_time * remaining_progress + random.randint(0, 2))
        return max(1, eta)

class BunchingDetector:
    """Detect bus bunching incidents"""
    
    @staticmethod
    def detect_bunching(buses: List[LiveBus], threshold_distance: float = 0.5) -> List[dict]:
        """
        Detect buses that are too close together (bunching)
        threshold_distance in km
        """
        bunching_incidents = []
        
        for i, bus1 in enumerate(buses):
            for j, bus2 in enumerate(buses[i+1:], i+1):
                if bus1.route_id == bus2.route_id:
                    distance = BunchingDetector.calculate_distance(
                        bus1.current_lat, bus1.current_lng,
                        bus2.current_lat, bus2.current_lng
                    )
                    
                    if distance < threshold_distance:
                        bunching_incidents.append({
                            "bus_1": bus1.bus_id,
                            "bus_2": bus2.bus_id,
                            "route_id": bus1.route_id,
                            "distance_km": round(distance, 2),
                            "severity": "high" if distance < 0.2 else "medium"
                        })
        
        return bunching_incidents
    
    @staticmethod
    def calculate_distance(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
        """Calculate distance between two GPS points in km"""
        # Haversine formula
        R = 6371  # Earth's radius in km
        
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lng = math.radians(lng2 - lng1)
        
        a = (math.sin(delta_lat / 2) ** 2 + 
             math.cos(lat1_rad) * math.cos(lat2_rad) * 
             math.sin(delta_lng / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
        return R * c