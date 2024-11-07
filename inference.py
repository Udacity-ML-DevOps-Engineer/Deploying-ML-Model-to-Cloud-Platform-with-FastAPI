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

def get_from_api(url="https://deploying-ml-model-to-cloud-platform.onrender.com/"):
    """
    Send GET request to the API and get prediction
    
    Args:
        params (dict): Input parameters for prediction
        url (str): API endpoint URL
    
    Returns:
        tuple: (prediction result, status code)
    """
    try:
        response = requests.get(url)
        return response.json(), response.status_code
    except requests.exceptions.RequestException as e:
        print(f"Error occurred: {e}")
        return None, 500

if __name__ == "__main__":
    # Sample data
    sample_data_1 = {
        "age": 29,
        "workclass": "Private",
        "fnlgt": 185908,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "Black",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 55,
        "native-country": "United-States"
    }
    
    sample_data_2 = {
        "age": 27,
        "workclass": "Private",
        "fnlgt": 160178,
        "education": "Some-college",
        "education-num": 10,
        "marital-status": "Divorced",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Female",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 38,
        "native-country": "United-States"
    }
    
    # GET request
    result_get, status_code_get = get_from_api()
    print(f"GET Status Code: {status_code_get}")
    print(f"GET Result: {result_get}\n")

    # POST request
    result_post, status_code_post = post_to_api(sample_data_1)
    print(f"POST Status Code: {status_code_post}")
    print(f"POST Input Data: {sample_data_1}")
    print(f"POST Prediction: {result_post}\n")
    
    # POST request
    result_get, status_code_get = post_to_api(sample_data_2)
    print(f"POST Status Code: {status_code_get}")
    print(f"POST Input Data: {sample_data_2}")
    print(f"POST Prediction: {result_get}")