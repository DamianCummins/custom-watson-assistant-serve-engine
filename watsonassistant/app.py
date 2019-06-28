"""Custom Watson Assistant Serve Engine.

This module can be deployed as a custom wrapper around the Watson Assistant service, to enable
the Watson OpenScale service to discover and score the assistant. Watson OpenScale does not support
out of the box integration with Watson Assistant at this time

The Flask app implements the Custom deployment endpoints specification documented here:
https://aiopenscale-custom-deployement-spec.mybluemix.net/

"""
from flask import Flask
from ibm_watson import AssistantV2

import os
import configparser
import flask
import numpy as np
import requests
import json


app = Flask(__name__, static_url_path='')

config = configparser.ConfigParser()
config.read('config.ini')

assistant = AssistantV2(
    iam_apikey = config['ASSISTANT']['APIKEY'], 
    version = '2019-02-28')
assistant.set_http_config({'timeout': 100})

assistant_id = 'ef7829c6-3fb8-4885-9a26-0485c29c0b0f'

def convert_output(output_data):
    """convert_output
    Args:
        output_data (List): The list of responses from Watson Assistant
    
    Returns:
        A dict in the format expected by Watson OpenScale
    """
    values = []
    labels = [
        'Cancel', 
        'Customer_Care_Appointments',
        'Customer_Care_Store_Hours',
        'Customer_Care_Store_Location',
        'General_Connect_to_Agent',
        'General_Greetings',
        'Goodbye',
        'Help',
        'Thanks'
    ]

    for response in output_data:
        probabilities = np.zeros(len(labels))
        intents = response['output']['intents']
        for intent in intents:
            probabilities[labels.index(intent['intent'])] = intent['confidence']
        values.append([intent['intent'], probabilities.tolist()])

    openscale_fields = ['intent', 'probabilities']
    openscale_values = values

    return {'fields': openscale_fields, 'labels': labels, 'values': openscale_values}

def convert_input(input_data):
    """convert_input
    Args:
        input_data (dict): The incoming scoring request from Watson OpenScale
    
    Returns:
        A List of values
    """
    openscale_values = input_data['values']
    return openscale_values

@app.route('/v1/deployments/assistant/message', methods=['POST'])
def send_message():
    if flask.request.method == "POST":
        payload = flask.request.get_json()

        response = []
        openscale_output = {}
        print(payload)

        # Create session.
        session_id = assistant.create_session(
            assistant_id = assistant_id
        ).get_result()['session_id']

        assistant.message(
            assistant_id,
            session_id
        )
        if payload is not None:
            messages = convert_input(payload)
            for message in messages:
                response.append(assistant.message(
                    assistant_id,
                    session_id,
                    input = { 'text': message[0] }
                ).get_result())
            openscale_output = convert_output(response)

        # We're done, so we delete the session.
        assistant.delete_session(
            assistant_id = assistant_id,
            session_id = session_id
        )
    return flask.jsonify(openscale_output)

@app.route('/v1/deployments', methods=['GET'])
def get_deployments():
    response = {}

    if flask.request.method == 'GET':
        host_url = flask.request.host_url
        response = {
            'count': 1,
            'resources': [
                {
                    'metadata': {
                        'guid': 'assistant',
                        'created_at': '2019-06-27T12:00:00Z',
                        'modified_at': '2019-06-27T13:00:00Z'
                    },
                    'entity': {
                        'name': 'Watson Assistant',
                        'description': 'Watson Assistant deployment',
                        'scoring_url': '{}v1/deployments/assistant/message'.format(host_url),
                        'asset': {
                            "name": "Watson Assistant",
                            "guid": "assistant"
                        },
                        'asset_properties': {
                            'problem_type': 'multiclass',
                            'input_data_type': 'unstructured_text'
                        }
                    }
                }
            ]
        }
        return flask.jsonify(response)

if __name__ == '__main__':
    port = os.getenv('PORT', '5000')
    app.run(host='0.0.0.0', port=int(port))