from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
try:
    from bus_data_reader import data_manager
    from models import BusRoute, BusStop, PassengerData, OptimizationResult
except ImportError as e:
    print(f"Import warning: {e}")
    print("Starting with basic functionality...")
    
    # Create a minimal data manager for testing
    class MinimalDataManager:
        def get_all_routes(self):
            return [{"route_id": 1, "route_name": "Test Route", "city": "Mumbai"}]
        
        def get_routes_by_city(self, city):
            return [{"route_id": 1, "route_name": f"Test Route in {city}", "city": city}]
        
        def get_stops_for_route(self, route_id):
            return [{"stop_id": 1, "stop_name": "Test Stop", "route_id": route_id}]
        
        def get_passenger_data_for_route(self, route_id, date=None):
            return [{"route_id": route_id, "boarding": 25, "alighting": 5}]
        
        def simulate_live_buses(self, route_id):
            return [{"bus_id": f"BUS_{route_id}_01", "status": "running"}]
        
        def get_optimization_comparison(self, route_id):
            return {
                "before": {"avg_wait_time": 12.5},
                "after": {"avg_wait_time": 7.8},
                "improvements": {"wait_time_reduction": 37.6}
            }
        
        def get_peak_hours_analysis(self, route_id):
            return {"boarding": {8: 45, 18: 52}, "occupancy": {8: 65, 18: 78}}
        
        def get_system_summary(self):
            return {"total_routes": 3, "cities": ["Mumbai", "Bangalore"]}
    
    data_manager = MinimalDataManager()
from typing import List, Optional
import uvicorn

app = FastAPI(title="Smart Bus Optimization API", version="1.0.0")
# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """API welcome endpoint"""
    return {
        "message": "🚌 Smart Bus Optimization API",
        "status": "running",
        "version": "1.0.0",
        "endpoints": {
            "routes": "/api/routes",
            "city_routes": "/api/routes/{city}",
            "stops": "/api/stops/{route_id}",
            "passenger_data": "/api/passenger-data/{route_id}",
            "live_buses": "/api/live-buses/{route_id}",
            "optimization": "/api/optimization/{route_id}",
            "peak_hours": "/api/peak-hours/{route_id}",
            "summary": "/api/summary"
        }
    }

@app.get("/api/routes")
async def get_all_routes():
    """Get all bus routes"""
    try:
        routes = data_manager.get_all_routes()
        return {
            "status": "success",
            "count": len(routes),
            "routes": routes
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/routes/{city}")
async def get_routes_by_city(city: str):
    """Get routes for a specific city"""
    try:
        routes = data_manager.get_routes_by_city(city)
        return {
            "status": "success",
            "city": city.title(),
            "count": len(routes),
            "routes": routes
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stops/{route_id}")
async def get_stops_for_route(route_id: int):
    """Get all stops for a route"""
    try:
        stops = data_manager.get_stops_for_route(route_id)
        return {
            "status": "success",
            "route_id": route_id,
            "count": len(stops),
            "stops": stops
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/passenger-data/{route_id}")
async def get_passenger_data(route_id: int, date: Optional[str] = None):
    """Get passenger data for a route"""
    try:
        data = data_manager.get_passenger_data_for_route(route_id, date)
        return {
            "status": "success",
            "route_id": route_id,
            "date_filter": date,
            "count": len(data),
            "passenger_data": data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/live-buses/{route_id}")
async def get_live_buses(route_id: int):
    """Get live bus positions for a route"""
    try:
        buses = data_manager.simulate_live_buses(route_id)
        return {
            "status": "success",
            "route_id": route_id,
            "timestamp": "2024-01-15 10:30:00",
            "buses": buses
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/optimization/{route_id}")
async def get_optimization_comparison(route_id: int):
    """Get before vs after optimization comparison"""
    try:
        comparison = data_manager.get_optimization_comparison(route_id)
        return {
            "status": "success",
            "route_id": route_id,
            "comparison": comparison
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/peak-hours/{route_id}")
async def get_peak_hours(route_id: int):
    """Get peak hours analysis for a route"""
    try:
        peak_data = data_manager.get_peak_hours_analysis(route_id)
        return {
            "status": "success",
            "route_id": route_id,
            "peak_analysis": peak_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/summary")
async def get_summary():
    """Get system summary"""
    try:
        summary = data_manager.get_system_summary()
        return {
            "status": "success",
            "summary": summary
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/alerts")
async def get_system_alerts():
    """Get system alerts"""
    alerts = [
        {
            "id": 1,
            "type": "bunching",
            "route_id": 1,
            "message": "Bus bunching detected on Route 1",
            "severity": "high",
            "timestamp": "2024-01-15 08:30:00",
            "action": "Rescheduling next bus departure"
        },
        {
            "id": 2,
            "type": "delay",
            "route_id": 4,
            "message": "Route 4 running behind schedule",
            "severity": "medium", 
            "timestamp": "2024-01-15 08:28:00",
            "action": "Adjusting frequency"
        }
    ]
    
    return {
        "status": "success",
        "count": len(alerts),
        "alerts": alerts
    }

if __name__ == "__main__":
    print("🚌 Starting Smart Bus Optimization API...")
    print("📍 API will be available at: http://localhost:8000")
    print("📖 API docs at: http://localhost:8000/docs")
    
    uvicorn.run(
        "app:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True
    )