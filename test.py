import requests

url = 'http://127.0.0.1:5000/get_similarity'
data = {
    'text1': 'I am not an idiot',
    'text2': 'I am an idiot'
}

response = requests.post(url, json=data)
print(response.json())
