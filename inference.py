import requests
import json

def post_to_api(data, url="https://deploying-ml-model-to-cloud-platform.onrender.com/predict"):
    """
    Send POST request to the API and get prediction
    
    Args:
        data (dict): Input data for prediction
        url (str): API endpoint URL
    
    Returns:
        tuple: (prediction result, status code)
    """
    try:
        response = requests.post(url, json=data)
        return response.json(), response.status_code
    except requests.exceptions.RequestException as e:
        print(f"Error occurred: {e}")
        return None, 500

if __name__ == "__main__":
    # Sample data
    sample_data = {
        "age": 39,
        "workclass": "State-gov",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital-gain": 2174,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States"
    }
    
    result, status_code = post_to_api(sample_data)
    print(f"Status Code: {status_code}")
    print(f"Prediction: {result}")