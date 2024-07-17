import numpy as np

# hyperparameters
weights = np.array([0.5,0.48,-0.7])
alpha = 0.1

# representation of streetlights into a matrix (on=1, off=0)
streetlights = np.array([
    [1,0,1],
    [0,1,1],
    [0,0,1],
    [1,1,1],
    [0,1,1],
    [1,0,1]
])

# lossless representation of outcome (stop=0,walk=1)
walk_or_stop = np.array([
    [0],
    [1],
    [0],
    [1],
    [1],
    [0]
])

# building the neural network
# move 40 times trough the graph and update weights
for iteration in range(40):
    error_for_all_lights = 0

    # going trough the graph
    for row_index in range(len(walk_or_stop)):
        # the 2 layers
        input = streetlights[row_index]
        goal_prediction = walk_or_stop[row_index]
        
        # calculating outcome and loss
        prediction = input.dot(weights)

        error = (goal_prediction - prediction) ** 2
        error_for_all_lights += error

        delta = prediction - goal_prediction
        weights = weights - (alpha * (input * delta))
        print("Prediction:" + str(prediction))
    print("Error:" + str(error) + " Prediction" + str(prediction)) 