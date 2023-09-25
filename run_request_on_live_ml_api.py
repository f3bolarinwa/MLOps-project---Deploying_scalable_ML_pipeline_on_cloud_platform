"""
Script to run sample request on API

Author: Femi Bolarinwa
Date: July 2023
"""

import requests
import json

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

#GET request
r = requests.get('https://femis-ml-pipeline-app.onrender.com/')
assert r.status_code == 200
print('Response code = {}'.format(r.status_code))
print('Response body = {}'.format(r.json()))


#POST request
r = requests.post('https://femis-ml-pipeline-app.onrender.com/inference', json=data) #data=json.dumps(data))
#r = requests.post('https://femis-ml-pipeline-app.onrender.com/docs#/default/run_inference_inference', json=data)
#r = requests.get('https://femis-ml-pipeline-app.onrender.com')

#assert r.status_code == 200

# print("Response code: %s" % r.status_code)
# print("Response body: %s" % r.json())



print('Response code = {}'.format(r.status_code))
print('Response body = {}'.format(r))#.json()))




'''

data = {
"age": 52,
"workclass": "Self-emp-not-inc",
"fnlgt": 209642,
"education": "HS-grad",
"marital_status": "Married-civ-spouse",
"occupation": "Exec-managerial",
"relationship": "Husband",
"race": "White",
"sex": "Male",
"hoursPerWeek": 45,
"nativeCountry": "United-States"
}
r = requests.post('https://salaryclassapp.herokuapp.com/inference', json=data)

assert r.status_code == 200

print("Response code: %s" % r.status_code)
print("Response body: %s" % r.json())

'''