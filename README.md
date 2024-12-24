## ANN implementation in C++

This project demonstrates the implementation of a feedforward Artificial Neural Network (ANN) in C++ for training and evaluating on the MNIST dataset. The network is designed to solve a multiclass classification problem, accurately classifying input images into their corresponding integer labels (0-9).

The network consists of three layers:

1. **Input Layer**: The input layer consists of 28x28 neurons, each representing a pixel of an MNIST image. These pixels are flattened into a 1D vector of 784 values, which are then fed into the hidden layer.

2. **Hidden Layer**: The network has one hidden layer with 128 neurons. Each of these neurons uses the ReLU (Rectified Linear Unit) activation function.

3. **Output Layer**: The output layer consists of 10 neurons, corresponding to the 10 possible digits (0-9) in the MNIST dataset. A softmax activation function is used for the output layer, which converts the raw outputs (logits) into probabilities.

The network is trained using backpropagation, where the error is propagated back through the network to adjust weights and biases, minimizing the loss. The evaluation of the trained model is done using accuracy, comparing predicted labels against true labels on a separate test set, ensuring the network's effectiveness at classifying new data.

### MNIST Dataset

The MNIST dataset consists of labeled images of handwritten digits (0-9), commonly used for training and testing machine learning algorithms. It is split into 60,000 training examples and 10,000 test examples.

The MNIST files are not my property. If used in a paper, I urge you to cite the authors.

More information is available on the official [MNIST website](https://yann.lecun.com/exdb/mnist/).


### Third-Party Code

This project uses the following third-party components:

- [`MNIST reader`](https://github.com/wichtounet/mnist) by Baptiste Wicht, licensed under the MIT License. See the license notice in `include/mnist/LICENSE` for details.

### License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for the full text.
