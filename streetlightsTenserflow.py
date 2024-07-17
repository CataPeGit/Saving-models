import tensorflow as tf
import numpy as np

"""
CREATING THE MODEL
"""

# Hyperparameters
alpha = 0.1

# Starting data
streetlights = np.array([
    [1, 0, 1],
    [0, 1, 1],
    [0, 0, 1],
    [1, 1, 1],
    [0, 1, 1],
    [1, 0, 1]
], dtype=np.float32)

walk_or_stop = np.array([
    [0],
    [1],
    [0],
    [1],
    [1],
    [0]
], dtype=np.float32)

# Initial weights - tensorflow variable so it is tracked and updated during training
weights = tf.Variable([[0.5], [0.48], [-0.7]], dtype=tf.float32)

# Training loop
for iteration in range(40):
    error_for_all_lights = 0
    for row_index in range(len(walk_or_stop)):
        input_data = streetlights[row_index:row_index+1]
        goal_prediction = walk_or_stop[row_index:row_index+1]

        # use tf to record and make the operations (gradient)
        with tf.GradientTape() as tape:
            prediction = tf.matmul(input_data, weights)
            error = tf.reduce_mean(tf.square(goal_prediction - prediction))

        gradients = tape.gradient(error, [weights])
        weights.assign_sub(alpha * gradients[0])

        error_for_all_lights += error.numpy()
        print(f"Prediction: {prediction.numpy()[0][0]}")
    print(f"Error: {error_for_all_lights}")
    print(f"Iteration {iteration} is done!\n\n")

print("Final weights:", weights.numpy())


"""
Define and convert model
"""
# Define a concrete function for the model
@tf.function
def model(inputs):
    return tf.matmul(inputs, weights)

# Convert the model to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_concrete_functions([model.get_concrete_function(tf.TensorSpec(shape=[None, 3], dtype=tf.float32))])
tflite_model = converter.convert()

# Save the TFLite model
with open('simple_model.tflite', 'wb') as f:
    f.write(tflite_model)


"""
Quantize the model
"""
converter.optimizations = [tf.lite.Optimize.DEFAULT]
quantized_tflite_model = converter.convert()

with open('quantized_simple_model.tflite', 'wb') as f:
    f.write(quantized_tflite_model)


