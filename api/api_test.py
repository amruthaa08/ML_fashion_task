import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__)))
root_dir = os.path.dirname(os.path.dirname(__file__))
from main import app
import base64
from io import BytesIO

from fastapi.testclient import TestClient
client=TestClient(app)
def test_valid_image():
    # with open('../sample_data/6836.jpg', 'rb') as image_file:
    with open(os.path.join(root_dir, "sample_data/6836.jpg"), 'rb') as image_file:
        files = {"file": ('valid_image', image_file, "image/jpeg")}
        response = client.post("http://127.0.0.1:8080/predict", files=files)
        assert response.status_code == 200

def test_invalid_image():
    # with open('../sample_data/sample.txt', 'rb') as image_file:
    with open(os.path.join(root_dir, "sample_data/sample.txt"), 'rb') as image_file:
        files = {"file": ('invalid_image', image_file)}
        response = client.post("http://127.0.0.1:8080/predict", files=files)
        assert response.status_code == 400
