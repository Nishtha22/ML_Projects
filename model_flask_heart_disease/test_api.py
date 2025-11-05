import requests
import json

# API endpoint
BASE_URL = "http://localhost:5000"

def test_health():
    """Test health endpoint"""
    print("Testing /health endpoint...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}\n")

def test_features():
    """Test features endpoint"""
    print("Testing /features endpoint...")
    response = requests.get(f"{BASE_URL}/features")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")

def test_prediction():
    """Test prediction endpoint"""
    print("Testing /predict endpoint...")
    
    # Sample patient data
    sample_data = {
        "Age": 56.0,
        "Gender": "Male",
        "Blood Pressure": 153.0,
        "Cholesterol Level": 155.0,
        "Exercise Habits": "High",
        "Smoking": "Yes",
        "Family Heart Disease": "Yes",
        "Diabetes": "No",
        "BMI": 24.99,
        "High Blood Pressure": "Yes",
        "Low HDL Cholesterol": "Yes",
        "High LDL Cholesterol": "No",
        "Alcohol Consumption": "High",
        "Stress Level": "Medium",
        "Sleep Hours": 7.63,
        "Sugar Consumption": "Medium",
        "Triglyceride Level": 342.0,
        "Fasting Blood Sugar": 120.0,
        "CRP Level": 12.97,
        "Homocysteine Level": 12.39
    }
    
    response = requests.post(
        f"{BASE_URL}/predict",
        json=sample_data,
        headers={'Content-Type': 'application/json'}
    )
    
    print(f"Status Code: {response.status_code}")
    print(f"Request Data: {json.dumps(sample_data, indent=2)}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")

def test_multiple_predictions():
    """Test multiple prediction samples"""
    print("Testing multiple predictions...")
    
    test_samples = [
        {
            "Age": 69.0,
            "Gender": "Female",
            "Blood Pressure": 146.0,
            "Cholesterol Level": 286.0,
            "Exercise Habits": "High",
            "Smoking": "No",
            "Family Heart Disease": "Yes",
            "Diabetes": "Yes",
            "BMI": 25.22,
            "High Blood Pressure": "No",
            "Low HDL Cholesterol": "Yes",
            "High LDL Cholesterol": "No",
            "Alcohol Consumption": "Medium",
            "Stress Level": "High",
            "Sleep Hours": 8.74,
            "Sugar Consumption": "Medium",
            "Triglyceride Level": 133.0,
            "Fasting Blood Sugar": 157.0,
            "CRP Level": 9.36,
            "Homocysteine Level": 19.30
        },
        {
            "Age": 32.0,
            "Gender": "Female",
            "Blood Pressure": 122.0,
            "Cholesterol Level": 293.0,
            "Exercise Habits": "High",
            "Smoking": "Yes",
            "Family Heart Disease": "Yes",
            "Diabetes": "No",
            "BMI": 24.13,
            "High Blood Pressure": "Yes",
            "Low HDL Cholesterol": "No",
            "High LDL Cholesterol": "Yes",
            "Alcohol Consumption": "Low",
            "Stress Level": "High",
            "Sleep Hours": 5.25,
            "Sugar Consumption": "High",
            "Triglyceride Level": 293.0,
            "Fasting Blood Sugar": 94.0,
            "CRP Level": 12.51,
            "Homocysteine Level": 5.96
        }
    ]
    
    for i, sample in enumerate(test_samples, 1):
        print(f"\nSample {i}:")
        response = requests.post(
            f"{BASE_URL}/predict",
            json=sample,
            headers={'Content-Type': 'application/json'}
        )
        print(f"Response: {json.dumps(response.json(), indent=2)}")

if __name__ == "__main__":
    try:
        print("=" * 60)
        print("Heart Disease Prediction API Test Suite")
        print("=" * 60 + "\n")
        
        test_health()
        test_features()
        test_prediction()
        test_multiple_predictions()
        
        print("\n" + "=" * 60)
        print("All tests completed!")
        print("=" * 60)
        
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to API. Make sure the server is running on http://localhost:5000")
    except Exception as e:
        print(f"Error: {e}")