import requests

url = "http://127.0.0.1:5000/predict"

# ── Test Case 1: High Value Loyalist ──
payload_1 = {
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
    "campaigns_accepted": 2,
    # ← ADD THESE:
    "marital_status": "Partnered",
    "education_group": "Postgraduate"
}

print("=" * 60)
print("TEST 1: High Value Loyalist Profile")
print("=" * 60)
response = requests.post(url, json=payload_1)
print("Status:", response.status_code)
print("Response:", response.json())

print("\n")

# ── Test Case 2: Budget Conscious Family ──
payload_2 = {
    "income": 30000,
    "recency": 50,
    "age": 50,
    "total_children": 3,
    "tenure_days": 300,
    "mnt_wines": 50,
    "mnt_fruits": 20,
    "mnt_meat": 80,
    "mnt_fish": 10,
    "mnt_sweet": 5,
    "mnt_gold": 10,
    "num_web_purchases": 2,
    "num_store_purchases": 3,
    "num_catalog": 0,
    "num_deals": 2,
    "num_web_visits": 8,
    "campaigns_accepted": 0,
    "marital_status": "Single",
    "education_group": "Undergraduate"
}

print("=" * 60)
print("TEST 2: Budget Conscious Family Profile")
print("=" * 60)
response = requests.post(url, json=payload_2)
print("Status:", response.status_code)
print("Response:", response.json())

print("\n")

# ── Test Case 3: Middle Class Active ──
payload_3 = {
    "income": 55000,
    "recency": 30,
    "age": 55,
    "total_children": 1,
    "tenure_days": 450,
    "mnt_wines": 200,
    "mnt_fruits": 50,
    "mnt_meat": 150,
    "mnt_fish": 50,
    "mnt_sweet": 30,
    "mnt_gold": 40,
    "num_web_purchases": 5,
    "num_store_purchases": 6,
    "num_catalog": 2,
    "num_deals": 1,
    "num_web_visits": 6,
    "campaigns_accepted": 1,
    "marital_status": "Widow",
    "education_group": "Other"
}

print("=" * 60)
print("TEST 3: Middle Class Active Profile")
print("=" * 60)
response = requests.post(url, json=payload_3)
print("Status:", response.status_code)
print("Response:", response.json())

print("\n")

# ── Test Case 4: Missing Field (Error Handling) ──
payload_4_invalid = {
    "income": 50000,
    "recency": 20,
    # ← Missing most fields
    "marital_status": "Partnered"
}

print("=" * 60)
print("TEST 4: Missing Fields (Error Handling)")
print("=" * 60)
response = requests.post(url, json=payload_4_invalid)
print("Status:", response.status_code)
print("Response:", response.json())

print("\n")

# ── Test Case 5: Invalid Category ──
payload_5_invalid = {
    "income": 50000,
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
    "campaigns_accepted": 2,
    "marital_status": "InvalidCategory",  # ← WRONG!
    "education_group": "Postgraduate"
}

print("=" * 60)
print("TEST 5: Invalid Category (Error Handling)")
print("=" * 60)
response = requests.post(url, json=payload_5_invalid)
print("Status:", response.status_code)
print("Response:", response.json())

print("\n")

# ── Test Case 6: Out of Range ──
payload_6_invalid = {
    "income": 500000,  # ← TOO HIGH (max 200000)
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
    "campaigns_accepted": 2,
    "marital_status": "Partnered",
    "education_group": "Postgraduate"
}

print("=" * 60)
print("TEST 6: Out of Range Value (Error Handling)")
print("=" * 60)
response = requests.post(url, json=payload_6_invalid)
print("Status:", response.status_code)
print("Response:", response.json())

print("\n" + "=" * 60)
print("✅ All tests completed!")
print("=" * 60)