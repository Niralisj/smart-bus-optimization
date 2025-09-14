// src/services/api.js
import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

// Create axios instance with base configuration
const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// API service functions
export const apiService = {
  // Live data endpoints
  getLiveGPS: async () => {
    try {
      const response = await api.get('/api/live/gps');
      return response.data;
    } catch (error) {
      console.error('Error fetching live GPS:', error);
      return { status: 'error', gps_data: [] };
    }
  },

  getLivePredictions: async (routeId = 'R1') => {
    try {
      const response = await api.get(`/api/live/predictions/${routeId}`);
      return response.data;
    } catch (error) {
      console.error('Error fetching predictions:', error);
      return { status: 'error', predictions: [] };
    }
  },

  getLiveAlerts: async () => {
    try {
      const response = await api.get('/api/alerts/live');
      return response.data;
    } catch (error) {
      console.error('Error fetching alerts:', error);
      return { status: 'error', live_alerts: [] };
    }
  },

  getLiveBoarding: async () => {
    try {
      const response = await api.get('/api/live/boarding');
      return response.data;
    } catch (error) {
      console.error('Error fetching boarding data:', error);
      return { status: 'error', boarding_events: [] };
    }
  },

  getOptimizationComparison: async (routeId = 'R1') => {
    try {
      const response = await api.get(`/api/optimization/comparison/${routeId}`);
      return response.data;
    } catch (error) {
      console.error('Error fetching comparison:', error);
      return { status: 'error' };
    }
  },

  getHackathonRequirements: async () => {
    try {
      const response = await api.get('/api/hackathon/requirements');
      return response.data;
    } catch (error) {
      console.error('Error fetching requirements:', error);
      return { status: 'error' };
    }
  },

  // System control
  controlSimulation: async (action) => {
    try {
      const response = await api.get(`/api/live/simulation/${action}`);
      return response.data;
    } catch (error) {
      console.error('Error controlling simulation:', error);
      return { status: 'error' };
    }
  },

  // Health check
  healthCheck: async () => {
    try {
      const response = await api.get('/');
      return response.data;
    } catch (error) {
      console.error('API health check failed:', error);
      return { status: 'error', message: 'Backend not available' };
    }
  }
};

export default apiService;