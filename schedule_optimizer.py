# schedule_optimizer.py
import json
import math
from datetime import datetime, timedelta
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)

class ScheduleOptimizer:
    """
    Smart Schedule Optimization Engine for Bus Routes
    Prevents bus bunching and optimizes frequency based on demand predictions
    """
    
    def __init__(self):
        self.route_configs = {
            'R001': {'base_frequency': 10, 'capacity': 50, 'min_frequency': 5, 'max_frequency': 20},
            'R002': {'base_frequency': 12, 'capacity': 45, 'min_frequency': 6, 'max_frequency': 25},
            'R003': {'base_frequency': 8, 'capacity': 60, 'min_frequency': 4, 'max_frequency': 15},
            'R004': {'base_frequency': 15, 'capacity': 40, 'min_frequency': 8, 'max_frequency': 30}
        }
        
        # Default config for routes not specified
        self.default_config = {'base_frequency': 10, 'capacity': 50, 'min_frequency': 5, 'max_frequency': 20}
        
        # Store optimization history for before/after comparison
        self.optimization_history = {}
        
        print("🚀 Schedule Optimizer initialized")
    
    def get_route_config(self, route_id: str) -> Dict:
        """Get configuration for a specific route"""
        return self.route_configs.get(route_id, self.default_config)
    
    def calculate_optimal_frequency(self, route_id: str, predicted_demand: int, current_hour: int) -> Dict[str, Any]:
        """
        Calculate optimal bus frequency based on predicted demand
        
        Args:
            route_id: Route identifier
            predicted_demand: Predicted passenger count
            current_hour: Current hour (0-23)
            
        Returns:
            Dict with optimal frequency, buses needed, and reasoning
        """
        config = self.get_route_config(route_id)
        base_freq = config['base_frequency']
        capacity = config['capacity']
        min_freq = config['min_frequency']
        max_freq = config['max_frequency']
        
        # Calculate utilization ratio
        utilization = predicted_demand / capacity if capacity > 0 else 0
        
        # Apply time-based adjustments
        rush_hour_multiplier = self._get_rush_hour_multiplier(current_hour)
        adjusted_demand = predicted_demand * rush_hour_multiplier
        
        # Calculate optimal frequency
        if adjusted_demand <= 10:  # Very low demand
            optimal_freq = min(max_freq, base_freq * 1.5)
            buses_needed = 1
            reason = "Low demand - reduced frequency"
        elif adjusted_demand <= 25:  # Low demand
            optimal_freq = base_freq
            buses_needed = 2
            reason = "Normal demand - standard frequency"
        elif adjusted_demand <= 40:  # Medium demand
            optimal_freq = max(min_freq, base_freq * 0.8)
            buses_needed = 3
            reason = "Medium demand - slightly increased frequency"
        elif adjusted_demand <= 60:  # High demand
            optimal_freq = max(min_freq, base_freq * 0.6)
            buses_needed = 4
            reason = "High demand - increased frequency"
        else:  # Very high demand
            optimal_freq = min_freq
            buses_needed = 5
            reason = "Peak demand - maximum frequency"
        
        # Ensure frequency is within bounds
        optimal_freq = max(min_freq, min(max_freq, optimal_freq))
        
        optimization_result = {
            "route_id": route_id,
            "predicted_demand": predicted_demand,
            "utilization_ratio": round(utilization, 2),
            "original_frequency": base_freq,
            "optimized_frequency": round(optimal_freq, 1),
            "buses_needed": buses_needed,
            "reasoning": reason,
            "capacity": capacity,
            "hour": current_hour,
            "rush_hour_factor": rush_hour_multiplier,
            "frequency_change": round(((base_freq - optimal_freq) / base_freq) * 100, 1),
            "timestamp": datetime.now().isoformat()
        }
        
        # Store for comparison
        self.optimization_history[route_id] = optimization_result
        
        return optimization_result
    
    def _get_rush_hour_multiplier(self, hour: int) -> float:
        """Get demand multiplier based on time of day"""
        if 7 <= hour <= 9:  # Morning rush
            return 1.4
        elif 17 <= hour <= 19:  # Evening rush
            return 1.5
        elif 12 <= hour <= 14:  # Lunch time
            return 1.1
        elif 22 <= hour or hour <= 5:  # Late night/early morning
            return 0.6
        else:  # Regular hours
            return 1.0
    
    def detect_bus_bunching(self, bus_positions: List[Dict]) -> List[Dict]:
        """
        Detect bus bunching on routes
        
        Args:
            bus_positions: List of bus position data
            
        Returns:
            List of bunching alerts
        """
        bunching_alerts = []
        
        # Group buses by route
        route_buses = {}
        for bus in bus_positions:
            route_id = bus.get('route_id', 'unknown')
            if route_id not in route_buses:
                route_buses[route_id] = []
            route_buses[route_id].append(bus)
        
        # Check for bunching on each route
        for route_id, buses in route_buses.items():
            if len(buses) < 2:
                continue
                
            # Sort buses by position (assuming position is a number representing progress)
            sorted_buses = sorted(buses, key=lambda x: x.get('position', 0))
            
            for i in range(len(sorted_buses) - 1):
                bus1 = sorted_buses[i]
                bus2 = sorted_buses[i + 1]
                
                # Calculate distance between buses
                pos_diff = abs(bus2.get('position', 0) - bus1.get('position', 0))
                
                # If buses are too close (less than 10% of route progress)
                if pos_diff < 0.1:  # 10% threshold
                    alert = {
                        "type": "bus_bunching",
                        "route_id": route_id,
                        "bus_ids": [bus1.get('bus_id'), bus2.get('bus_id')],
                        "severity": "high" if pos_diff < 0.05 else "medium",
                        "message": f"Buses {bus1.get('bus_id')} and {bus2.get('bus_id')} are bunching on route {route_id}",
                        "recommendation": "Adjust departure time of trailing bus by 3-5 minutes",
                        "timestamp": datetime.now().isoformat()
                    }
                    bunching_alerts.append(alert)
        
        return bunching_alerts
    
    def generate_optimization_summary(self, route_id: str = None) -> Dict[str, Any]:
        """
        Generate before/after optimization comparison
        
        Args:
            route_id: Specific route to analyze, or None for all routes
            
        Returns:
            Optimization summary with improvements
        """
        if route_id and route_id in self.optimization_history:
            data = self.optimization_history[route_id]
            
            # Calculate improvements
            original_wait_time = data['original_frequency'] / 2  # Average wait time
            optimized_wait_time = data['optimized_frequency'] / 2
            wait_time_improvement = ((original_wait_time - optimized_wait_time) / original_wait_time) * 100
            
            # Calculate efficiency metrics
            original_buses_per_hour = 60 / data['original_frequency']
            optimized_buses_per_hour = 60 / data['optimized_frequency']
            efficiency_change = ((optimized_buses_per_hour - original_buses_per_hour) / original_buses_per_hour) * 100
            
            return {
                "route_id": route_id,
                "optimization_type": "single_route",
                "original_schedule": {
                    "frequency_minutes": data['original_frequency'],
                    "buses_per_hour": round(original_buses_per_hour, 1),
                    "average_wait_time": round(original_wait_time, 1)
                },
                "optimized_schedule": {
                    "frequency_minutes": data['optimized_frequency'],
                    "buses_per_hour": round(optimized_buses_per_hour, 1),
                    "average_wait_time": round(optimized_wait_time, 1)
                },
                "improvements": {
                    "wait_time_change_percent": round(wait_time_improvement, 1),
                    "efficiency_change_percent": round(efficiency_change, 1),
                    "buses_needed": data['buses_needed'],
                    "utilization_ratio": data['utilization_ratio']
                },
                "reasoning": data['reasoning'],
                "timestamp": data['timestamp']
            }
        
        else:
            # Return summary for all routes
            total_routes = len(self.optimization_history)
            if total_routes == 0:
                return {"message": "No optimizations performed yet"}
            
            avg_wait_improvement = 0
            avg_efficiency_improvement = 0
            
            for route_data in self.optimization_history.values():
                original_wait = route_data['original_frequency'] / 2
                optimized_wait = route_data['optimized_frequency'] / 2
                wait_improvement = ((original_wait - optimized_wait) / original_wait) * 100
                avg_wait_improvement += wait_improvement
            
            avg_wait_improvement /= total_routes
            
            return {
                "optimization_type": "system_wide",
                "total_routes_optimized": total_routes,
                "system_improvements": {
                    "average_wait_time_reduction": round(avg_wait_improvement, 1),
                    "routes_with_improved_efficiency": total_routes,
                    "total_optimizations_today": total_routes
                },
                "latest_optimization": datetime.now().isoformat()
            }
    
    def get_optimization_recommendations(self, route_data: Dict) -> List[Dict]:
        """
        Get actionable recommendations for route optimization
        
        Args:
            route_data: Current route performance data
            
        Returns:
            List of optimization recommendations
        """
        recommendations = []
        
        route_id = route_data.get('route_id', 'unknown')
        current_demand = route_data.get('current_passengers', 0)
        predicted_demand = route_data.get('predicted_demand', 0)
        
        # High demand recommendation
        if predicted_demand > 50:
            recommendations.append({
                "type": "frequency_increase",
                "priority": "high",
                "message": f"Increase frequency for route {route_id} - predicted demand: {predicted_demand}",
                "action": "Deploy additional bus",
                "estimated_improvement": "25% reduction in wait time"
            })
        
        # Low utilization recommendation
        elif predicted_demand < 15:
            recommendations.append({
                "type": "frequency_decrease",
                "priority": "medium",
                "message": f"Reduce frequency for route {route_id} - low predicted demand: {predicted_demand}",
                "action": "Extend interval by 5 minutes",
                "estimated_improvement": "15% cost savings"
            })
        
        # Capacity optimization
        utilization = current_demand / 50  # Assuming 50 seat capacity
        if utilization > 0.9:
            recommendations.append({
                "type": "capacity_alert",
                "priority": "high",
                "message": f"Route {route_id} approaching capacity - {int(utilization*100)}% full",
                "action": "Dispatch additional bus",
                "estimated_improvement": "Prevent overcrowding"
            })
        
        return recommendations

# Global optimizer instance
optimizer = ScheduleOptimizer()

def get_optimizer():
    """Get the global optimizer instance"""
    return optimizer

def optimize_route_schedule(route_id: str, predicted_demand: int, current_hour: int = None) -> Dict[str, Any]:
    """
    Optimize schedule for a specific route
    
    Args:
        route_id: Route identifier
        predicted_demand: Predicted passenger demand
        current_hour: Current hour (defaults to current time)
        
    Returns:
        Optimization result
    """
    if current_hour is None:
        current_hour = datetime.now().hour
    
    return optimizer.calculate_optimal_frequency(route_id, predicted_demand, current_hour)

def get_system_optimization_summary() -> Dict[str, Any]:
    """Get system-wide optimization summary"""
    return optimizer.generate_optimization_summary()

def get_route_optimization_summary(route_id: str) -> Dict[str, Any]:
    """Get optimization summary for specific route"""
    return optimizer.generate_optimization_summary(route_id)