# Smart Bus Optimization Challenge

## 📌 Problem Statement
Urban bus systems in Indian Tier-1 cities (e.g., Bangalore, Delhi, Pune) rely on **static timetables** that fail to adapt to real-world conditions.  
This leads to:
- Bus bunching (multiple buses arriving together)  
- Under-utilized trips during off-peak hours  
- Unpredictable passenger wait times  

Transit agencies **lack tools** to forecast demand surges and adjust schedules in real time.
 

##Prototype built in 36 hours for Hackathon — **Problem Statement 2**  

Urban bus systems in Tier-1 Indian cities often run on static timetables. This causes **bus bunching, empty off-peak trips, and unpredictable wait times**.  
Our solution: a **Smart Bus Management System** that adapts in real time to improve efficiency and passenger experience.  

---

## Features  

   **Data ingestion** → uses multiple CSVs 
   **Real-time simulation** → buses move with mocked GPS + live passenger counts  
   **Scheduling engine** → reschedules delayed buses, dispatches extras if overcrowded  
   **Prediction model** → forecasts ridership for upcoming hours  
  **Alerts** → detects delays, overcrowding, and notifies in real time  
  **Dashboard/UI** → shows optimized vs original schedules, ridership charts, alerts, and live bus map  

---

## 🛠️ Tech Stack  

- **Backend**: FastAPI (Python)  
- **Frontend**: HTML, JavaScript (Chart.js, Leaflet.js)  
- **Data/ML**: Pandas, Scikit-learn / basic time series  
- **Database**: SQLite (for prototype)  

---

## 🚀 Quickstart  

```bash
# 1. Clone repo
git clone https://github.com/your-username/smart-bus-optimization.git
cd smart-bus-optimization

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate   # Mac/Linux
.\venv\Scripts\activate    # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run backend
uvicorn backend.app:app --reload



