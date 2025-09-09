
import pandas as pd
import requests
from io import StringIO

def generate_data():
    """
    Downloads the Pima Indians Diabetes dataset and saves it as 'diabetes_dataset.csv'.
    """
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        # Column names for the dataset
        column_names = [
            "Pregnancies", "Glucose", "BloodPressure", "SkinThickness", 
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
        ]
        
        # Read the CSV data from the response content
        data = pd.read_csv(StringIO(response.text), header=None, names=column_names)
        
        # Save the dataset to a CSV file
        data.to_csv("diabetes_dataset.csv", index=False)
        
        print("Dataset downloaded and saved as 'diabetes_dataset.csv'")
        
    except requests.exceptions.RequestException as e:
        print(f"Error downloading the dataset: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    generate_data()
