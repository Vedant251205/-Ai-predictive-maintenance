import csv
import random
import os
from datetime import datetime, timedelta

def generate_data():
    machines = ['Rotary_Kiln_01', 'Ball_Mill_02', 'Vertical_Roller_Mill_01', 'Clinker_Cooler_02', 'Coal_Mill_01']
    data = []
    
    # Start time (May 1st, 2026)
    start_time = datetime(2026, 5, 1)
    
    # Set random seed for reproducibility
    random.seed(42)
    
    for i in range(100):
        # Generate readings every 6 hours
        timestamp = (start_time + timedelta(hours=i * 6)).strftime('%Y-%m-%d %H:%M:%S')
        
        # Select machine randomly
        machine = random.choice(machines)
        
        # Generate operational parameters with 15% chance of maintenance requirement
        is_anomaly = random.random() < 0.15
        
        if is_anomaly:
            # Anomaly parameters (e.g. overheating, high vibration, high pressure, low oil level)
            temp = round(random.uniform(120.0, 165.0), 2)
            vibration = round(random.uniform(7.2, 12.5), 2)
            pressure = round(random.uniform(5.2, 7.8), 2)
            rpm = random.randint(1450, 1700)
            oil = round(random.uniform(25.0, 50.0), 2)
            maintenance = 1
        else:
            # Normal operating parameters
            temp = round(random.uniform(65.0, 95.0), 2)
            vibration = round(random.uniform(1.2, 3.9), 2)
            pressure = round(random.uniform(2.2, 4.5), 2)
            rpm = random.randint(950, 1050)
            oil = round(random.uniform(75.0, 95.0), 2)
            maintenance = 0
            
        data.append([timestamp, machine, temp, vibration, pressure, rpm, oil, maintenance])
        
    return data

def main():
    # Ensure dataset directory exists
    os.makedirs('dataset', exist_ok=True)
    csv_file_path = os.path.join('dataset', 'machine_data.csv')
    
    headers = [
        'Timestamp', 
        'Machine_ID', 
        'Temperature_C', 
        'Vibration_mm_s', 
        'Pressure_bar', 
        'Speed_RPM', 
        'Oil_Level_Percent', 
        'Maintenance_Required'
    ]
    
    with open(csv_file_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(generate_data())
        
    print(f"Synthetic dataset for UltraTech Cement predictive maintenance successfully created at: {csv_file_path}")

if __name__ == "__main__":
    main()
