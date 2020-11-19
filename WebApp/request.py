import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'GRE':2, 'TOFEL':9, 'CGPA':6})

print(r.json())