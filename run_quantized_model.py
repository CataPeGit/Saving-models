# Lint as: python3
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Run inference and measure time using PyCoral and basic Python."""

import time
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters import common

# Define the model path
model_path = "quantized_streetlight_model.tflite"

# Initialize the Edge TPU interpreter
interpreter = make_interpreter(model_path)
interpreter.allocate_tensors()

# Function definitions for interacting with the interpreter
def output_tensor(interpreter, i):
    """Gets a model's ith output tensor."""
    return interpreter.tensor(interpreter.get_output_details()[i]['index'])()

def input_details(interpreter, key):
    """Gets a model's input details by specified key."""
    return interpreter.get_input_details()[0][key]

def input_size(interpreter):
    """Gets a model's input size as (width, height) tuple."""
    _, height, width, _ = input_details(interpreter, 'shape')
    return width, height

def input_tensor(interpreter):
    """Gets a model's input tensor view as list of lists of lists."""
    tensor_index = input_details(interpreter, 'index')
    return interpreter.tensor(tensor_index)()

def set_input(interpreter, data):
    """Copies data to a model's input tensor."""
    input_tensor(interpreter)[:, :] = data

# Prepare multiple input data examples
input_data_examples = [
    [1, 0, 1],
    [0, 1, 1],
    [0, 0, 1],
    [1, 1, 1],
    [0, 1, 1],
    [1, 0, 1]
]

# Run inference on each example and measure the time
for input_data in input_data_examples:
    # Convert input data to a format suitable for Edge TPU
    input_data_bytearray = bytearray(input_data)
    set_input(interpreter, input_data_bytearray)
    
    # Measure inference time
    start_time = time.time()
    interpreter.invoke()
    end_time = time.time()
    
    # Get the output tensor
    output_data = output_tensor(interpreter, 0)
    inference_time = end_time - start_time
    
    # Print results
    print(f"Input: {input_data}")
    print(f"Prediction: {output_data}")
    print(f"Inference Time: {inference_time} seconds\n\n")
