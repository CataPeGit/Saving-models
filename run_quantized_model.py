import time
import numpy as np
import tflite_runtime.interpreter as tflite
from PIL import Image

# Path to the quantized TensorFlow Lite model
model_path = 'quantized_streetlight_model.tflite'

# Initialize the TensorFlow Lite interpreter
interpreter = tflite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

# Example inputs (similar to streetlights data)
inputs = [
    [1, 0, 1],  # Similar to streetlights[0]
    [0, 1, 1],  # Similar to streetlights[1]
    [0, 0, 1],  # Similar to streetlights[2]
    [1, 1, 1],  # Similar to streetlights[3]
    [0, 1, 1],  # Similar to streetlights[4]
    [1, 0, 1]   # Similar to streetlights[5]
]

# Prepare input and output arrays (int8)
input_data = np.array(inputs, dtype=np.int8)

# Perform inference for each input
for idx, input_data in enumerate(inputs):
    print(f"Input {idx + 1}: {input_data}")

    # Start timer
    start_time = time.time()

    # Perform inference
    interpreter.set_tensor(input_details['index'], input_data.reshape(input_details['shape']))
    interpreter.invoke()
    output = interpreter.get_tensor(output_details['index'])

    # End timer
    elapsed_time = time.time() - start_time

    # Print the output
    print(f"Predicted value: {output[0]}")
    print(f"Inference time: {elapsed_time:.5f} seconds\n")

# Clean up resources
interpreter.close()
