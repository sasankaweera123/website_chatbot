import pytest
from flask.testing import FlaskClient
from app import app


@pytest.fixture
def client() -> FlaskClient:
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


def test_predict(client: FlaskClient):
    response = client.post('/predict', json={'message': 'Hi'})
    assert response.status_code == 200
    assert 'message' in response.json
    assert response.json['message'] == "Hey :-)" or "Hello, how can I help you?" or "Hi there, how can I help?" or \
           "Hello, what can I do for you?" or "Hi there, how can I help?"
