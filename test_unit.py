import pytest
import joblib
from fastapi.testclient import TestClient
from Backend.main import app

client = TestClient(app)

@pytest.fixture
def my_image():
    image = 'images_Test/image copy 4.png'
    return image

def test_model():
    model = None
    model = joblib.load('ML/Model/Model.dump')
    assert model != None

def test_prediction(my_image):
    with open(my_image, "rb") as f:
        response = client.post(
            "/Prediction",
            files={"file": ("test.png", f, "image/png")}
        )

    assert response.status_code == 200