import pandas as pd
import numpy as np
import random
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

def train_delay_model():
    """
    Simulates transportation data and trains a Random Forest model
    to predict delay probability based on congestion and mode.
    Also exports the simulated data to 'transportation_data.xlsx'.
    """
    np.random.seed(42)
    random.seed(42)
    n_samples = 1000

    # --- 1. Generate Realistic Logistics Data (12 Columns) ---
    shipment_ids = [f"SHP-{1000+i}" for i in range(n_samples)]
    dates = pd.date_range(start="2025-01-01", periods=n_samples, freq="H").strftime("%Y-%m-%d").tolist()
    
    origins = ['Mumbai', 'Delhi', 'Chennai', 'Kolkata', 'Bangalore', 'Hyderabad', 'Pune', 'Ahmedabad']
    destinations = ['Mumbai', 'Delhi', 'Chennai', 'Kolkata', 'Bangalore', 'Hyderabad', 'Pune', 'Ahmedabad']
    
    origin_col = [random.choice(origins) for _ in range(n_samples)]
    dest_col = []
    for o in origin_col:
        d = random.choice(destinations)
        while d == o: # Ensure origin != dest
            d = random.choice(destinations)
        dest_col.append(d)
        
    distances = np.random.randint(200, 2500, n_samples)
    
    # Core Features
    congestion = np.random.uniform(0, 100, n_samples) # 0-100
    # Mode: 0=Road, 1=Rail, 2=Coastal
    # Let's assign text labels too
    modes_text = []
    modes_encoded = []
    for _ in range(n_samples):
        r = random.random()
        if r < 0.6: m, c = 'Road', 0
        elif r < 0.9: m, c = 'Rail', 1
        else: m, c = 'Coastal', 2
        modes_text.append(m)
        modes_encoded.append(c)
        
    volume = np.random.uniform(10, 500, n_samples)
    
    # Additional Context
    weather_impact = np.random.uniform(0, 1, n_samples) # 0=Clear, 1=Storm
    driver_ratings = np.random.randint(1, 6, n_samples) # 1-5 stars
    vehicle_types = [random.choice(['Truck-20T', 'Truck-40T']) if m == 'Road' else ( 'Wagon-Box' if m == 'Rail' else 'Container-Ship') for m in modes_text]
    
    # Target: Delay Probability Calculation (Synthetic Logic)
    delay_prob = np.zeros(n_samples)
    
    for i in range(n_samples):
        base_delay = 0.05
        mode_factor = 0
        
        # Road is sensitive to congestion and weather
        if modes_encoded[i] == 0: 
            mode_factor = (congestion[i] / 100) * 0.7 + (weather_impact[i] * 0.2)
        # Rail is stable but capacity constrained (simulated by volume impact)
        elif modes_encoded[i] == 1: 
            mode_factor = (congestion[i] / 100) * 0.15 + 0.05 
        # Coastal is sensitive to weather
        elif modes_encoded[i] == 2: 
            mode_factor = (congestion[i] / 100) * 0.1 + (weather_impact[i] * 0.6)
            
        noise = np.random.normal(0, 0.02)
        prob = base_delay + mode_factor + noise
        delay_prob[i] = np.clip(prob, 0, 1)

    # DataFrame Construction
    df = pd.DataFrame({
        'Shipment_ID': shipment_ids,
        'Date': dates,
        'Origin': origin_col,
        'Destination': dest_col,
        'Distance_KM': distances,
        'Mode': modes_text,
        'Vehicle_Type': vehicle_types,
        'Volume_Tons': np.round(volume, 2),
        'Congestion_Index': np.round(congestion, 1),
        'Weather_Severity_Index': np.round(weather_impact, 2),
        'Driver_Rating': driver_ratings,
        'Predicted_Delay_Risk': np.round(delay_prob, 4)
    })
    
    # Export to Excel with Data Dictionary
    try:
        # Create Data Dictionary DataFrame
        data_dict = pd.DataFrame([
            {"Field": "Shipment_ID", "Description": "Unique identifier for the shipment log", "Role in Code": "Context/Identifier (Not used in training)"},
            {"Field": "Date", "Description": "Date of the shipment record", "Role in Code": "Context/Reporting (Not used in training)"},
            {"Field": "Origin", "Description": "Starting city of the transport", "Role in Code": "Context/Routing Info (Not used in training)"},
            {"Field": "Destination", "Description": "Ending city of the transport", "Role in Code": "Context/Routing Info (Not used in training)"},
            {"Field": "Distance_KM", "Description": "Distance between origin and destination", "Role in Code": "Context/Cost Analysis (Not used in training)"},
            {"Field": "Mode", "Description": "Transport mode (Road, Rail, Coastal)", "Role in Code": "Converted to 'Mode_Encoded' feature (0,1,2) for the Random Forest model."},
            {"Field": "Vehicle_Type", "Description": "Specific vehicle/vessel configuration", "Role in Code": "Context/Resource Allocation (Not used in training)"},
            {"Field": "Volume_Tons", "Description": "Weight of the cargo in tons", "Role in Code": "Feature: 'Volume'. Used to train model on capacity-related delay risks."},
            {"Field": "Congestion_Index", "Description": "Real-time traffic/route saturation (0-100)", "Role in Code": "Feature: 'Congestion'. Primary driver for delay prediction in the AI model."},
            {"Field": "Weather_Severity_Index", "Description": "Impact of weather (0.0=Clear, 1.0=Severe)", "Role in Code": "Used in synthetic generation logic to influence delay probability, especially for Coastal/Road."},
            {"Field": "Driver_Rating", "Description": "Performance rating of driver (1-5)", "Role in Code": "Context/Quality Metric (Not used in training)"},
            {"Field": "Predicted_Delay_Risk", "Description": "Probability of delay (0.0 to 1.0)", "Role in Code": "Target Variable ('y'). This is what the AI model learns to predict."}
        ])

        with pd.ExcelWriter("transportation_data.xlsx", engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name="Shipment Data", index=False)
            data_dict.to_excel(writer, sheet_name="Field Descriptions", index=False)
            
        print("Data exported to transportation_data.xlsx with Data Dictionary.")
    except Exception as e:
        print(f"Could not export Excel: {e}")

    # Prepare Data for Training (Numeric only)
    X = pd.DataFrame({
        'Congestion': congestion,
        'Mode_Encoded': modes_encoded,
        'Volume': volume
    })
    # Target: We use the calculated probability as the regression target
    y = delay_prob
    
    # Generate Binary outcome for Classification Metrics (Delay vs No Delay)
    # Using the probability to sample an actual outcome
    y_binary = np.array([np.random.binomial(1, p) for p in y])

    # Split Data
    X_train, X_test, y_train, y_test, y_bin_train, y_bin_test = train_test_split(
        X, y, y_binary, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Return model and test data for evaluation
    return model, X_test, y_test, y_bin_test

def predict_delay_risk(model, congestion_level, volume, mode_type):
    """
    Predicts delay probability for a specific scenario.
    mode_type: 'Road', 'Rail', 'Coastal'
    """
    mode_map = {'Road': 0, 'Rail': 1, 'Coastal': 2}
    
    if mode_type not in mode_map:
        return 0.0
        
    mode_code = mode_map[mode_type]
    
    input_data = pd.DataFrame({
        'Congestion': [congestion_level],
        'Mode_Encoded': [mode_code],
        'Volume': [volume]
    })
    
    risk = model.predict(input_data)[0]
    return float(np.clip(risk, 0, 1))
