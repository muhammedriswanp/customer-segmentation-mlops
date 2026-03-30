import requests

url = "http://127.0.0.1:5000/predict"

payload = {
    "income": 75000,
    "recency": 20,
    "age": 45,
    "total_children": 1,
    "tenure_days": 400,
    "mnt_wines": 500,
    "mnt_fruits": 30,
    "mnt_meat": 200,
    "mnt_fish": 40,
    "mnt_sweet": 20,
    "mnt_gold": 60,
    "num_web_purchases": 6,
    "num_store_purchases": 5,
    "num_catalog": 3,
    "num_deals": 1,
    "num_web_visits": 4,
    "campaigns_accepted": 2
}

response = requests.post(url, json=payload)
print("Status:", response.status_code)
print("Response:", response.json())
