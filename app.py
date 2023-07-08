"""
This module contains a Flask application for a chatbot API.

The application provides two routes:
- GET '/' returns the base HTML template.
- POST '/predict' expects a JSON payload with a 'message' field and returns a response from the chatbot.

Usage:
    - Run this module to start the Flask application.
"""
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from chat import get_response

app = Flask(__name__)
CORS(app)


@app.get('/')
def index_get():
    """
        Handle the GET request for the '/' route.

        Returns:
            str: Rendered HTML template.
    """
    return render_template('base.html')


@app.post('/predict')
def predict():
    """
        Handle the POST request for the '/predict' route.

        Expects a JSON payload with a 'message' field.
        Returns a JSON response with a 'message' field containing the chatbot response.

        Returns:
            dict: JSON response with the chatbot response.
    """
    text = request.get_json().get('message')
    response = get_response(text)
    message = {'message': response}
    return jsonify(message)


if __name__ == '__main__':
    app.run(debug=True)
