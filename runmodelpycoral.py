from pycoral.adapters import classify
from pycoral.utils.edgetpu import make_interpreter

# Initialize the PyCoral interpreter
interpreter = make_interpreter("streetlightsTenserflow.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test input data
input_data = [[1.0, 0.0, 1.0]]  # Input data without using numpy array

# Resize input tensor to match model expectations
interpreter.resize_tensor_input(input_details[0]['index'], (1, len(input_data[0])))
interpreter.allocate_tensors()

# Set input tensor
interpreter.set_tensor(input_details[0]['index'], input_data)

# Run inference
interpreter.invoke()

# Get output tensor
output = interpreter.tensor(output_details[0]['index'])()

# Print prediction
print("Prediction:", output)
