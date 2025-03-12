import requests

url = "http://127.0.0.1:5000/predict"

data = {
    "features": [
    3.21, -5.68, 4.91, -2.73,
    7.45, -9.87, 10.32, -11.21,
    5.67, -8.44, 7.78, -6.32,
    9.56, -7.14, 8.29, -12.38,
    15.89, -14.32, 13.45, -10.67,
    11.21, -16.43, 17.34, -18.29,
    19.72, -20.56, 22.89, -24.14,29.90,-32.87
]
}

headers = {"Content-Type": "application/json"}
response = requests.post(url, json=data, headers=headers)

print(response.json())  # Should print {'prediction': 'Fraudulent' or 'Legitimate'}
