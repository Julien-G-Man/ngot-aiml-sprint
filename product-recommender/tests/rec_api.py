import requests
from fastapi.testclient import TestClient
from app.main import app

url = "http://localhost:8002"
client = TestClient(app)

health = requests.get(url+"/health")
h_data = health.json()
print(health.status_code, h_data.get("status"))


r = client.post("/recommend", json={'query': 'rainy day gear'})
r_data = r.json()
print(r.status_code, r_data.get("recommendations"))