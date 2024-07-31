# what is Keras?
- Keras is a high-level neural networks API, written in Python and capable of running on top of TensorFlow, CNTK, or Theano. It was developed with a focus on enabling fast experimentation. Being able to go from idea to result with the least possible delay is key to doing good research.
# our model input
- ballX, ballY, ballVX, ballVY, paddleY
# our model output
- paddleY

# notes
so we consider [ballX, ballY, ballVX, ballVY, paddleY] as training samples and paddleY when the ball reaches the paddle_x (-3) as the target value.
# model
- we will use a simple feedforward neural network with 3 layers.
- input layer: 5 neurons
- hidden layer: 10 neurons
- output layer: 1 neuron
- activation function: ReLU
- loss function: mean squared error
- optimizer: adam
- epochs: 1000
- batch size: 10
- learning rate: 0.001
- validation split: 0.2
- shuffle: True
- verbose: 1
- callbacks: early stopping, model checkpoint
# training
- we will train the model on 1000 samples.
- we will use 80% of the samples for training and 20% for validation.
- we will use early stopping to prevent overfitting.
- we will use model checkpoint to save the best model.
# collecting the data
- we will collect the data by playing the game and recording the input and output values.
## how to collect the data?
- collect random  ball position, and direction if dz is negative (the ball is moving towards the ai paddle) from `-GAME_WIDTH / 2` to `GAME_WIDTH / 2` as the training samples.
- collect the ball x position when the ball reaches the paddle_x (bz > -5.5 and z <-6)
- store the training samples and target values in a csv file.
