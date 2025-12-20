import pandas as pd
import io
import os

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Culture War Companies Data Cleaning ---
## Import Culture war companies data from Culture_War_Companies_160_fullmeta.csv
def import_culture_war_data(file_path):
    """
    Imports and cleans the Culture War Companies dataset.
    """
    # Convert relative path to absolute path based on script location
    if not os.path.isabs(file_path):
        file_path = os.path.join(SCRIPT_DIR, file_path)
    
    df = pd.read_csv(file_path)
    
    ## make "Event Date" a datetime object
    df['Event Date'] = pd.to_datetime(df['Event Date'], errors='coerce')
    
    return df

if __name__ == "__main__":
    # Call the function
    df = import_culture_war_data('Culture_War_Companies_160_fullmeta.csv')
    print(df)
    print(f"\nDataframe shape: {df.shape}")
    print(f"\nColumn names: {df.columns.tolist()}")

#--- END CULTURE WAR COMPANIES DATA CLEANING ---

#--- Create Data Dictionary ---

def create_data_dictionary():
    """Creates a dictionary of all datasets"""
    culture_war_data = import_culture_war_data('Culture_War_Companies_160_fullmeta.csv')
    
    full_dictionary = {
        "culturewardata": culture_war_data,
    }
    
    return full_dictionary