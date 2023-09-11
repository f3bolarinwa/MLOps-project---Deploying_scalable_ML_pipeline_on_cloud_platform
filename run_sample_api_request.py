"""
Script to run sample request on API

Author: Femi Bolarinwa
Date: July 2023
"""

import requests

data = {
    'age': 52,
    'workclass': 'Self-emp-not-inc',
    'fnlgt': 209642,
    'education': 'HS-grad',
    'marital_status': 'Married-civ-spouse',
    'occupation': 'Exec-managerial',
    'relationship': 'Husband',
    'race': 'White',
            'sex': 'Male',
            'hoursPerWeek': 45,
            'nativeCountry': 'United-States'
}

r = requests.post('http://127.0.0.1:8000/inference', json=data)

assert r.status_code == 200

# print("Response code: %s" % r.status_code)
# print("Response body: %s" % r.json())

print('Response code = {}'.format(r.status_code))
print('Response body = {}'.format(r.json()))
