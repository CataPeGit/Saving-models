"""
Run inference and measure time using PyCoral
"""

import time
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters import common

# Define the model path
model_path = "quantized_streetlight_model.tflite"

# Initialize the Edge TPU interpreter
interpreter = make_interpreter(model_path)
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = common.input_details(interpreter)
output_details = common.output_details(interpreter)

# Prepare multiple input data examples
input_data_examples = [
    [1, 0, 1],
    [0, 1, 1],
    [0, 0, 1],
    [1, 1, 1],
    [0, 1, 1],
    [1, 0, 1]
]

# Convert input data to int8 and add batch dimension
input_data_examples = [[[int8 for int8 in example]] for example in input_data_examples]  # Add batch dimension

# Run inference on each example and measure the time
for input_data in input_data_examples:
    input_data = bytearray(input_data[0][0])
    common.set_input(interpreter, input_data)
    
    start_time = time.time()
    interpreter.invoke()
    end_time = time.time()
    
    # Get the output
    output_data = common.output_tensor(interpreter, output_details[0]['index'])
    inference_time = end_time - start_time
    
    print(f"Input: {input_data}")
    print(f"Prediction: {output_data}")
    print(f"Inference Time: {inference_time} seconds\n\n")
