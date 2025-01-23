INPUT_SCHEMA = {
    "prompt_value": {
        'datatype': 'STRING',
        'required': True,
        'shape': [1],
        'example': ["Hello, how are you doing today?"]
    },
    "input_value": {
        'datatype': 'STRING',
        'required': True,
        'shape': [1],
        'example': ["Leela speaks in a high-pitched, fast-paced, and cheerful tone, full of energy and happiness. The recording is very high quality with no background noise."]
    }
}
IS_STREAMING_OUTPUT = True
