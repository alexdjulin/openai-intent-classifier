# -*- coding: utf-8 -*-
import argparse
import requests
from flask import Flask, request, jsonify, render_template
from intent_classifier import IntentClassifier
import logging
LOG = logging.getLogger('intent_classifier')

app = Flask(__name__)
model = IntentClassifier()


@app.route('/')
def index() -> str:
    '''
    Render index.html
    '''
    model_id = model.model_id
    intent_list = ', '.join([intent for intent in model.intents_list if intent != 'none'])
    return render_template('index.html', model_id=model_id, intent_list=intent_list)


@app.route('/ready')
def ready() -> tuple:
    '''
    Check if the model is ready

    Return:
        'OK', 200 if the model is ready
        'Not ready', 423 if the model is not ready
    '''

    if model.is_ready():
        LOG.debug('Model is ready.')
        return 'OK', 200
    else:
        LOG.debug('Model is not ready.')
        return 'Not ready', 423


@app.route('/intent', methods=['POST'])
def intent() -> tuple:
    '''
    POST request, predicts the intent of the text message.

    Returns:
        tuple: JSON response with predicted intent, response code

    Raises:
        400: Bad Request (body or 'text' key missing)
        423: Locked
        500: Internal Error

    '''

    LOG.debug(f'Received request to predict intent: {request.json}')

    # Call /ready and check if the model is ready
    try:
        response = requests.get(request.url_root + 'ready')
        if response.status_code != 200:
            raise Exception()
    except Exception:
        LOG.error('Model not ready. Check the model name-id.')
        return jsonify({"label": "MODEL_NOT_READY", "message": "Model is not ready, check the model name-id."}), 423

    # fetch the text from the request
    try:
        data = request.json
        if not data:
            raise Exception()

    except Exception:
        LOG.error('Request body missing.')
        return jsonify({"label": "BODY_MISSING", "message": "Request doesn't have a body."}), 400

    # fetch the text from the request
    try:
        text = data['text']

    except KeyError:
        LOG.error('Text missing from request body.')
        return jsonify({"label": "TEXT_MISSING", "message": "\"text\" missing from request body."}), 400

    # predict the corresponding list of intents
    try:
        result = model.predict_intent(text)
        print('RESULT', result)
        if result is None:
            raise Exception("Failed to predict intent. Unclear, inappropriate or out-of-context question?")

    except Exception as e:
        LOG.error(f"Error predicting intent: {e}")
        return jsonify({"label": "INTERNAL_ERROR", "message": str(e)}), 500

    # return the result as a JSON response to display
    LOG.debug(f'Predicted intent: {result}')
    return jsonify(result), 200


def main() -> None:
    '''
    Main function to run the Flask server.
    '''

    # parse input arguments
    arg_parser = argparse.ArgumentParser(description='Specify the model to use and the server port number.')
    arg_parser.add_argument('--model', '-m', type=str, required=True, help='Path to model name, directory or file.')
    arg_parser.add_argument('--port', '-p', type=int, default='8080', help='Server port number.')
    args = arg_parser.parse_args()

    # load model before running the flask server
    model.load(model=str(args.model), )

    # run flask server
    app.run(host='0.0.0.0', port=args.port)


if __name__ == '__main__':
    main()
