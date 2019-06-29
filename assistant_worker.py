"""assistant_worker.py
"""
from threading import Thread
import numpy as np

class AssistantWorker(Thread):
    """AssistantWorker(Thread)
    
    Args:
        Thread
    
    This class is a worker thread which, when run, is responsible for taking a list of messages
    from a queue, opening a Session with the Watson Assistant service, and sending each message 
    sequentially for classification.

    Each tuple in the queue contains:
        - chunk_id: an int which identifies the chunk of messages
        - messages: a list of message values
        - results: a list of results. The thread should update results[chunk_id] on completion

    """

    def __init__(self, queue, assistant, assistant_id):
        Thread.__init__(self)
        self.queue = queue
        self.assistant = assistant
        self.assistant_id = assistant_id

    def run(self):
        while True:
            # Take the message list from the queue 
            # and send to Watson Assistant
            chunk_id, messages, results = self.queue.get()
            print("Processing chunk_id", chunk_id)
            responses = []
            try:
                # Create session.
                session_id = self.assistant.create_session(
                    assistant_id = self.assistant_id
                ).get_result()['session_id']

                self.assistant.message(
                    self.assistant_id,
                    session_id
                )
                for message in messages:
                    responses.append(self.assistant.message(
                        self.assistant_id,
                        session_id,
                        input = { 'text': message[0] }
                    ).get_result())
                results[chunk_id] = self.convert_output(responses)
                
                # We're done, so we delete the session.
                self.assistant.delete_session(
                    assistant_id = self.assistant_id,
                    session_id = session_id
                )
            finally:
                print("chunk_id: ", chunk_id, " completed")
                self.queue.task_done()

    def convert_output(self, output_data):
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
