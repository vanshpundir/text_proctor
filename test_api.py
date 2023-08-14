import requests

url = 'http://127.0.0.1:5000/get_similarity'
data = {
    'text1': 'Hi, I am Vansh',
    'text2': 'this is not a proper valid sattement'
}

response = requests.post(url, json=data)

print(response.json())
#TODO:
# try with ddqnn,
# see euclidean distance,
# differnce between cosine and euclidean,
