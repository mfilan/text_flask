import json
from flask import Flask, jsonify, request
from transformers import pipeline
import logging
from flask_cors import CORS
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

app = Flask(__name__)
CORS(app)
PUBLIC = 'public'
app.logger.setLevel(logging.INFO)

@app.route(f'/{PUBLIC}/isAlive')
def is_alive():
    return 'Alive'


@app.route(f'/{PUBLIC}/predict', methods=['POST'])
def predict():
    text = request.json["text"]

    if text is None:
        return jsonify(error="JSON content is empty"), 400

    output = summarizer(text, max_length=130, min_length=30, do_sample=False)

    response = app.response_class(
        response=json.dumps(output),
        status=200,
        mimetype='application/json'
    )
    return response


if __name__ == '__main__':
    app.logger.info("App is starting ...")
    app.logger.info(
        f"Server is running, listening on port 8000")
    from waitress import serve

    serve(app, port=8080)