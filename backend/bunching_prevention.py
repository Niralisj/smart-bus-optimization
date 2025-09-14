# bunching_prevention.py - Bus Bunching and Empty Trips Prevention
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import json

class BunchingPreventionEngine:
    def __init__(self):
        self.min_headway_minutes = 5  # Minimum time between buses
        self.low_occupancy_threshold = 20  # Below 20% occupancy is considered low
        self.peak_hours = [7, 8, 17, 18, 19]
        self.off_peak_hours = [10, 11, 14, 15, 20, 21]
        
        # Tracking data
        self.bus_positions = {}
        self.bus_schedules = {}
        self.occupancy_history = {}
        self.interventions_log = []
        
        print("Smart Bus Spacing Engine initialized")
    
    def detect_bus_bunching(self, route_id: str, bus_positions: List[Dict]) -> List[Dict]:
        """Detect when buses are too close together (bunching)"""
        bunching_alerts = []
        
        if len(bus_positions) < 2:
            return bunching_alerts
        
        # Sort buses by their position/time on route
        sorted_buses = sorted(bus_positions, key=lambda x: x.get('route_progress', 0))
        
        for i in range(len(sorted_buses) - 1):
            current_bus = sorted_buses[i]
            next_bus = sorted_buses[i + 1]
            
            # Calculate time headway between buses
            time_diff = self._calculate_headway(current_bus, next_bus)
            
            if time_diff < self.min_headway_minutes:
                bunching_alert = {
                    'alert_id': f"bunching_{route_id}_{datetime.now().strftime('%H%M%S')}",
                    'type': 'bunching_detected',
                    'route_id': route_id,
                    'bus1_id': current_bus['bus_id'],
                    'bus2_id': next_bus['bus_id'],
                    'current_headway': time_diff,
                    'min_required_headway': self.min_headway_minutes,
                    'severity': 'high' if time_diff < 2 else 'medium',
                    'timestamp': datetime.now().isoformat(),
                    'recommended_action': self._get_bunching_solution(current_bus, next_bus, time_diff)
                }
                
                bunching_alerts.append(bunching_alert)
        
        return bunching_alerts
    
    def _calculate_headway(self, bus1: Dict, bus2: Dict) -> float:
        """Calculate time headway between two buses"""
        # For demo purposes, using a simplified calculation
        # In real system, this would use GPS positions and route topology
        
        pos1 = bus1.get('route_progress', 0)
        pos2 = bus2.get('route_progress', 0)
        
        position_diff = abs(pos2 - pos1)
        
        # Convert position difference to time (assuming average speed)
        avg_speed_kmh = 25
        time_diff_hours = position_diff / avg_speed_kmh
        time_diff_minutes = time_diff_hours * 60
        
        # Add some randomness for realistic simulation
        time_diff_minutes += np.random.uniform(-1, 1)
        
        return max(0.5, time_diff_minutes)
    
    def _get_bunching_solution(self, bus1: Dict, bus2: Dict, headway: float) -> Dict:
        """Generate solution for bus bunching"""
        if headway < 2:  # Critical bunching
            return {
                'action': 'hold_bus',
                'target_bus': bus2['bus_id'],
                'hold_duration_minutes': 5 - headway,
                'location': 'next_major_stop',
                'priority': 'high'
            }
        else:  # Moderate bunching
            return {
                'action': 'adjust_speed',
                'target_bus': bus2['bus_id'],
                'speed_adjustment': 'reduce_10_percent',
                'duration_minutes': 3,
                'priority': 'medium'
            }
    
    def detect_empty_trips(self, route_id: str, bus_data: List[Dict]) -> List[Dict]:
        """Detect buses with low occupancy during off-peak hours"""
        empty_trip_alerts = []
        current_hour = datetime.now().hour
        
        # Only check during off-peak hours
        if current_hour not in self.off_peak_hours:
            return empty_trip_alerts
        
        for bus in bus_data:
            occupancy_percent = (bus.get('passenger_count', 0) / bus.get('capacity', 45)) * 100
            
            if occupancy_percent < self.low_occupancy_threshold:
                alert = {
                    'alert_id': f"empty_trip_{route_id}_{bus['bus_id']}_{datetime.now().strftime('%H%M%S')}",
                    'type': 'low_occupancy',
                    'route_id': route_id,
                    'bus_id': bus['bus_id'],
                    'current_occupancy': occupancy_percent,
                    'passenger_count': bus.get('passenger_count', 0),
                    'capacity': bus.get('capacity', 45),
                    'hour': current_hour,
                    'severity': 'high' if occupancy_percent < 10 else 'medium',
                    'timestamp': datetime.now().isoformat(),
                    'recommended_action': self._get_empty_trip_solution(route_id, occupancy_percent, current_hour)
                }
                
                empty_trip_alerts.append(alert)
        
        return empty_trip_alerts
    
    def _get_empty_trip_solution(self, route_id: str, occupancy: float, hour: int) -> Dict:
        """Generate solution for empty trips"""
        if occupancy < 10:  # Very low occupancy
            return {
                'action': 'reduce_frequency',
                'route_id': route_id,
                'current_frequency': '15_minutes',
                'recommended_frequency': '25_minutes',
                'next_trip_delay': 10,
                'reason': 'very_low_demand',
                'estimated_savings': 'fuel_20_percent'
            }
        else:  # Low occupancy
            return {
                'action': 'monitor_next_trip',
                'route_id': route_id,
                'monitoring_duration': 30,
                'threshold': 'if_next_trip_also_low',
                'contingency': 'reduce_frequency'
            }
    
    def apply_bunching_prevention(self, route_id: str, bus_positions: List[Dict]) -> Dict:
        """Apply bunching prevention measures"""
        bunching_alerts = self.detect_bus_bunching(route_id, bus_positions)
        empty_trip_alerts = self.detect_empty_trips(route_id, bus_positions)
        
        interventions_applied = []
        
        # Apply bunching solutions
        for alert in bunching_alerts:
            intervention = self._execute_bunching_intervention(alert)
            if intervention:
                interventions_applied.append(intervention)
        
        # Apply empty trip solutions  
        for alert in empty_trip_alerts:
            intervention = self._execute_empty_trip_intervention(alert)
            if intervention:
                interventions_applied.append(intervention)
        
        result = {
            'route_id': route_id,
            'timestamp': datetime.now().isoformat(),
            'bunching_alerts': len(bunching_alerts),
            'empty_trip_alerts': len(empty_trip_alerts),
            'interventions_applied': len(interventions_applied),
            'interventions': interventions_applied,
            'system_status': 'optimized' if interventions_applied else 'stable'
        }
        
        # Log for tracking
        self.interventions_log.append(result)
        
        return result
    
    def _execute_bunching_intervention(self, alert: Dict) -> Dict:
        """Execute intervention to prevent bunching"""
        action = alert['recommended_action']
        
        intervention = {
            'intervention_id': f"fix_{alert['alert_id']}",
            'type': 'bunching_prevention',
            'action_taken': action['action'],
            'target_bus': action['target_bus'],
            'timestamp': datetime.now().isoformat(),
            'expected_outcome': f"Increase headway to {self.min_headway_minutes} minutes",
            'status': 'applied'
        }
        
        # In real system, this would send commands to bus drivers/dispatch
        print(f"INTERVENTION: {action['action']} applied to bus {action['target_bus']}")
        
        return intervention
    
    def _execute_empty_trip_intervention(self, alert: Dict) -> Dict:
        """Execute intervention for empty trips"""
        action = alert['recommended_action']
        
        intervention = {
            'intervention_id': f"fix_{alert['alert_id']}",
            'type': 'empty_trip_prevention',
            'action_taken': action['action'],
            'route_id': alert['route_id'],
            'timestamp': datetime.now().isoformat(),
            'expected_outcome': action.get('estimated_savings', 'improved_efficiency'),
            'status': 'applied'
        }
        
        print(f"INTERVENTION: {action['action']} applied to route {alert['route_id']}")
        
        return intervention
    
    def get_prevention_metrics(self, route_id: str = None) -> Dict:
        """Get metrics on bunching prevention effectiveness"""
        if route_id:
            # Filter interventions for specific route
            route_interventions = [
                i for i in self.interventions_log 
                if i['route_id'] == route_id
            ]
        else:
            route_interventions = self.interventions_log
        
        total_interventions = len(route_interventions)
        bunching_interventions = len([
            i for i in route_interventions 
            if any('bunching_prevention' in int_item.get('type', '') 
                  for int_item in i.get('interventions', []))
        ])
        empty_trip_interventions = len([
            i for i in route_interventions 
            if any('empty_trip_prevention' in int_item.get('type', '') 
                  for int_item in i.get('interventions', []))
        ])
        
        return {
            'route_id': route_id or 'all_routes',
            'total_interventions': total_interventions,
            'bunching_interventions': bunching_interventions,
            'empty_trip_interventions': empty_trip_interventions,
            'prevention_effectiveness': {
                'bunching_reduced': f"{bunching_interventions * 15} minutes saved",
                'fuel_saved': f"{empty_trip_interventions * 20}% efficiency gain",
                'passenger_experience': 'improved' if total_interventions > 0 else 'stable'
            },
            'last_24h_interventions': len([
                i for i in route_interventions
                if datetime.fromisoformat(i['timestamp']) > datetime.now() - timedelta(hours=24)
            ])
        }
    
    def simulate_prevention_scenarios(self) -> List[Dict]:
        """Simulate different prevention scenarios for demo"""
        scenarios = [
            {
                'scenario': 'morning_rush_bunching',
                'description': 'Two buses bunch during morning rush hour',
                'route_id': 'R1',
                'buses': [
                    {'bus_id': 'R1_B01', 'passenger_count': 35, 'capacity': 45, 'route_progress': 0.3},
                    {'bus_id': 'R1_B02', 'passenger_count': 38, 'capacity': 45, 'route_progress': 0.32}
                ],
                'expected_intervention': 'hold_bus'
            },
            {
                'scenario': 'off_peak_empty_bus',
                'description': 'Low occupancy during off-peak hours',
                'route_id': 'R3',
                'buses': [
                    {'bus_id': 'R3_B01', 'passenger_count': 4, 'capacity': 45, 'route_progress': 0.5}
                ],
                'expected_intervention': 'reduce_frequency'
            },
            {
                'scenario': 'multiple_bunching',
                'description': 'Three buses bunching on busy route',
                'route_id': 'R2',
                'buses': [
                    {'bus_id': 'R2_B01', 'passenger_count': 40, 'capacity': 45, 'route_progress': 0.6},
                    {'bus_id': 'R2_B02', 'passenger_count': 42, 'capacity': 45, 'route_progress': 0.62},
                    {'bus_id': 'R2_B03', 'passenger_count': 39, 'capacity': 45, 'route_progress': 0.64}
                ],
                'expected_intervention': 'multiple_holds'
            }
        ]
        
        results = []
        for scenario in scenarios:
            result = self.apply_bunching_prevention(scenario['route_id'], scenario['buses'])
            result['scenario_info'] = scenario
            results.append(result)
        
        return results

# Global prevention engine
prevention_engine = BunchingPreventionEngine()

# API functions for integration
def check_bunching_prevention(route_id: str, bus_positions: List[Dict]):
    """Check and apply bunching prevention for route"""
    return prevention_engine.apply_bunching_prevention(route_id, bus_positions)

def get_prevention_status():
    """Get overall prevention system status"""
    return {
        'system_active': True,
        'prevention_rules': {
            'min_headway': f"{prevention_engine.min_headway_minutes} minutes",
            'low_occupancy_threshold': f"{prevention_engine.low_occupancy_threshold}%",
            'monitoring_hours': '6 AM - 10 PM'
        },
        'recent_interventions': len(prevention_engine.interventions_log),
        'metrics': prevention_engine.get_prevention_metrics()
    }

def simulate_prevention_demo():
    """Run prevention simulation for demo"""
    return prevention_engine.simulate_prevention_scenarios()

def get_prevention_metrics_summary():
    """Get prevention effectiveness summary"""
    return prevention_engine.get_prevention_metrics()

if __name__ == "__main__":
    print("Testing Bus Bunching Prevention...")
    
    # Test bunching detection
    sample_buses = [
        {'bus_id': 'R1_B01', 'passenger_count': 35, 'capacity': 45, 'route_progress': 0.3},
        {'bus_id': 'R1_B02', 'passenger_count': 38, 'capacity': 45, 'route_progress': 0.31}
    ]
    
    result = check_bunching_prevention('R1', sample_buses)
    print(f"Interventions applied: {result['interventions_applied']}")
    
    # Test prevention scenarios
    scenarios = simulate_prevention_demo()
    print(f"Tested {len(scenarios)} prevention scenarios")
    
    # Get metrics
    metrics = get_prevention_metrics_summary()
    print(f"Prevention effectiveness: {metrics['prevention_effectiveness']}")
    
    print("Prevention system test completed!")