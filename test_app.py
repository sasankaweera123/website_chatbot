"""
This module contains unit tests for the Flask application.

The tests use the pytest testing framework and test the '/predict'
endpoint of the application.

Usage:
    - Run the tests using a test runner or the pytest command.
"""

import pytest
from flask.testing import FlaskClient
from app import app


@pytest.fixture
def client() -> FlaskClient:
    """
    Fixture to set up the Flask test client.

    Returns:
        FlaskClient: The Flask test client.
    """
    app.config['TESTING'] = True
    with app.test_client() as test_client:
        yield test_client


def test_predict(client: FlaskClient):
    """
    Test case for the '/predict' endpoint.

    Args:
        client (FlaskClient): The Flask test client.

    Raises:
        AssertionError: If any of the test assertions fail.
    """
    response = client.post('/predict', json={'message': 'Hi'})
    assert response.status_code == 200
    assert 'message' in response.json
