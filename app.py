"""Custom Watson Assistant Serve Engine.

This module can be deployed as a custom wrapper around the Watson Assistant service, to enable
the Watson OpenScale service to discover and score the assistant. Watson OpenScale does not support
out of the box integration with Watson Assistant at this time

The Flask app implements the Custom deployment endpoints specification documented here:
https://aiopenscale-custom-deployement-spec.mybluemix.net/

"""
from flask import Flask
from ibm_watson import AssistantV2
from queue import Queue
from time import time
from assistant_worker import AssistantWorker

import os
import configparser
import flask

app = Flask(__name__, static_url_path='')

config = configparser.ConfigParser()
config.read('config.ini')

assistant = AssistantV2(
    iam_apikey = config['ASSISTANT']['APIKEY'], 
    version = config['ASSISTANT']['VERSION'])
assistant.set_http_config({'timeout': int(config['ASSISTANT']['TIMEOUT'])})

assistant_id = config['ASSISTANT']['ASSISTANT_ID']

@app.route('/v1/deployments/assistant/message', methods=['POST'])
def send_message():
    """send_message

    Watson OpenScale will invoke this API to score the underlying model.
    The implementation needs to be able to handle a payload with a large number
    of values. For instance, during an explanation job, Watson OpenScale will 
    send a request containing ~5000 perturbed values.

    Method:
        POST

    Body:
        a JSON object in the format of a Watson OpenScale scoring payload

    Response:
        a JSON object in the format of a Watson OpenScale scoring response

    """
    ts = time()
    if flask.request.method == "POST":
        payload = flask.request.get_json()

        openscale_output = {
            'fields': [],
            'labels': [],
            'values': []
        }

        queue = Queue()
        # Create 50 AssistantWorker threads
        for x in range(50):
            worker = AssistantWorker(queue, assistant, assistant_id)
            # Setting daemon to True will let the main thread exit even though the workers are blocking
            worker.daemon = True
            worker.start()
        
        # Split up incoming messages into chunks of 100
        message_chunks = _chunks(payload['values'], 100)
        results = [[] for chunk in message_chunks]

        counter = 0
        # Put the tasks into the queue as a tuple
        for chunk in message_chunks:
            queue.put((counter, chunk, results))
            counter += 1

        # Causes the main thread to wait for the queue to finish processing all the tasks
        queue.join()
        print('Took %s', time() - ts)
        
        # Structure the response with the values in order
        for result in results:
            openscale_output['fields'] = result['fields']
            openscale_output['labels'] = result['labels']
            openscale_output['values'].extend(result['values'])

    return flask.jsonify(openscale_output)

@app.route('/v1/deployments', methods=['GET'])
def get_deployments():
    """get_deployments

    Watson OpenScale will invoke this API to discover and subscribe to the Watson Assistant deployment.
    The properties defined in the response will be used to correctly configure the subscription.

    Watson Assistant is defined as a multiclass problem type with unstructured text as an input.

    Method:
        GET

    Response:
        a JSON object in the format of a Watson OpenScale discover response

    """
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

def _chunks(l, n):
    return [l[i:i + n] for i in range(0, len(l), n)]

if __name__ == '__main__':
    port = os.getenv('PORT', '5000')
    app.run(host='0.0.0.0', port=int(port))