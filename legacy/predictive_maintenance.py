"""
UltraTech Cement - AI-Based Predictive Maintenance Project
Part 1: Data Exploration and Analysis

This Python script loads and inspects the machine health monitoring dataset.
Predictive maintenance helps detect anomalies in equipment like Rotary Kilns, 
Ball Mills, and Crushers before failure occurs, saving downtime and maintenance costs.

Author: AI Maintenance Predictor
"""

# Requirement 1: Import the pandas library
# Pandas is the most popular library in Python for data manipulation and analysis.
# We import it as 'pd' which is a standard industry convention to write shorter code.
import pandas as pd

def main():
    # Print a header for the application
    print("=" * 80)
    print("      ULTRATECH CEMENT - AI-BASED PREDICTIVE MAINTENANCE DASHBOARD")
    print("=" * 80)
    
    # Requirement 2: Load the dataset file named machine_data.csv from the dataset folder
    # We use pd.read_csv() to read our CSV (Comma Separated Values) file.
    # The path is 'dataset/machine_data.csv' because the file is inside the 'dataset' directory.
    dataset_path = "dataset/machine_data.csv"
    
    try:
        # Load the CSV data into a Pandas DataFrame (a 2D tabular data structure)
        df = pd.read_csv(dataset_path)
        print(f"\n[SUCCESS] Dataset successfully loaded from: {dataset_path}\n")
    except FileNotFoundError:
        print(f"\n[ERROR] The file '{dataset_path}' was not found.")
        print("Please run 'generate_dataset.py' first to generate the sample dataset.\n")
        return
    except Exception as e:
        print(f"\n[ERROR] An error occurred while loading the dataset: {e}\n")
        return

    # Requirement 3: Display the first 10 rows of the dataset
    # We use the df.head(10) method, which returns the first 10 rows of the DataFrame.
    # This is useful for getting a quick glimpse of what the actual data looks like.
    print("-" * 80)
    print("Requirement 3: Displaying the first 10 rows of the dataset")
    print("-" * 80)
    print(df.head(10))
    print("\n" + "=" * 80 + "\n")

    # Requirement 4: Display dataset columns
    # df.columns returns the names of all the columns in our DataFrame.
    # We convert it to a list and print it to see what features (variables) we have.
    print("-" * 80)
    print("Requirement 4: Displaying the dataset columns")
    print("-" * 80)
    columns_list = list(df.columns)
    print("The columns present in the machine health dataset are:")
    for idx, col in enumerate(columns_list, 1):
        print(f"  {idx}. {col}")
    print("\n" + "=" * 80 + "\n")

    # Requirement 5: Display dataset information
    # df.info() displays a concise summary of the DataFrame, including:
    # - The total number of rows (entries)
    # - Column names and count
    # - Number of non-null (valid) values in each column
    # - Data types (e.g., float64, int64, object) of each column
    # - Memory usage
    print("-" * 80)
    print("Requirement 5: Displaying the dataset information (Structure & Data Types)")
    print("-" * 80)
    df.info()
    print("\n" + "=" * 80 + "\n")

    # Requirement 6: Display statistical summary
    # df.describe() generates descriptive statistics that summarize the central tendency,
    # dispersion, and shape of a dataset's distribution.
    # It shows count, mean, standard deviation, minimum, maximum, and percentiles for numeric columns.
    print("-" * 80)
    print("Requirement 6: Displaying the statistical summary of numeric columns")
    print("-" * 80)
    print(df.describe())
    print("=" * 80)
    
    # Quick insight summarizing the data for Cement Plant Operators
    print("\n[AI INSIGHTS FOR ULTRATECH CEMENT OPERATORS]")
    total_records = len(df)
    failures = df['Maintenance_Required'].sum()
    failure_rate = (failures / total_records) * 100
    
    print(f"-> Total Machine Readings Analyzed: {total_records}")
    print(f"-> Machines Requiring Immediate Maintenance: {failures} ({failure_rate:.1f}%)")
    print(f"-> Average Operating Temperature: {df['Temperature_C'].mean():.2f} C")
    print(f"-> Maximum Operating Temperature Recorded: {df['Temperature_C'].max():.2f} C")
    print(f"-> Average Machine Vibration: {df['Vibration_mm_s'].mean():.2f} mm/s")
    print("=" * 80)

if __name__ == "__main__":
    main()
