## ANN implementation in C++

This project demonstrates the implementation of a feedforward Artificial Neural Network (ANN) in C++ for training and evaluating on the MNIST dataset. The network is designed to solve a multiclass classification problem, accurately classifying input images into their corresponding integer labels (0-9). No machine learning libraries were used, with the only third-party component being a dataset loader.

### Network Architecture

1. **Input Layer**: The input layer consists of 28x28 neurons, each representing a pixel of an MNIST image. These pixels are flattened into a 1D vector of 784 values, which are then fed into the hidden layer.

2. **Hidden Layer**: The network has one hidden layer with 128 neurons. Each of these neurons uses the ReLU (Rectified Linear Unit) activation function.

3. **Output Layer**: The output layer consists of 10 neurons, corresponding to the 10 possible digits (0-9) in the MNIST dataset. A softmax activation function is used for the output layer, which converts the raw outputs into probabilities.

The network is trained using backpropagation, where the error is propagated back through the network to adjust weights and biases, minimizing the loss.

### MNIST Dataset

The MNIST dataset consists of labeled images of handwritten digits (0-9), commonly used for training and testing machine learning algorithms. It is split into 60,000 training examples and 10,000 test examples.

The MNIST files are not my property. If used in a paper, I urge you to cite the authors.

More information is available on the official [MNIST website](https://yann.lecun.com/exdb/mnist/).

### Third-Party Code

This project uses the following third-party components:

- [`MNIST dataset reader`](https://github.com/wichtounet/mnist) by Baptiste Wicht, licensed under the MIT License. See the license notice in `include/mnist/LICENSE` for details.

### Evaluation Methods

The model is evaluated using several metrics that provide a comprehensive view of its behavior during training and testing.

1. **Sparse Cross-Entropy Loss**: Measures how well the predicted probabilities align with the true class label in a multiclass classification problem. It is computed as the negative logarithm of the predicted probability for the true class. If the model assigns a higher probability to the correct class, the loss is smaller, which indicates a better prediction.

2. **Accuracy**: Measures how correctly the model is predicting the true labels. It is computed by dividing the number of correct predictions by the total number of predictions.

3. **Confusion Matrix**: Visualizes the model's performance across different classes. It is a table where:

- Diagonal elements represent correct predictions (true positives).
- Off-diagonal elements represent misclassifications (false positives and false negatives).

4. **Class Precision**: Precision helps to measure how many of the predicted instances for each class are actually correct. High precision indicates that false positives are minimized.

### Model Evaluation Metrics

The model was trained for 10 epochs (iterations over the entire training dataset), with each epoch taking approximately 63,000 milliseconds (81,000 ms without parallel computing). With learning rate of 0.001, the following metrics were obtained:

- `avg.loss: 0.100203`
- `accuracy: 96.93%`
- `Confusion matrix:`

   <img width="492" alt="Screenshot 2024-12-25 at 19 18 48" src="https://github.com/user-attachments/assets/72891254-0e15-4fe6-ae20-d908b104c21d" />

- `Class precision:`

  <img width="107" alt="Screenshot 2024-12-25 at 19 20 08" src="https://github.com/user-attachments/assets/7446875b-1d61-496c-a19c-ce02623863f9" />

The average loss of `0.100203` and accuracy of `96.93%` indicate that the model is performing well and generalizing effectively to unseen data.

In the confusion matrix, true positives dominate, and misclassifications are rare, demonstrating that the model's predictions are generally correct. The most frequent misclassification occurred when the model misclassified images of the number 9 as the number 7.

Class precision remains high across all classes, with values ranging from 0.95 to 0.99, suggesting that the model is reliably predicting the correct labels for each class.

### Performance Enhancements

Even though the evaluation metrics are good, further optimizations could enhance the model's performance. Below are some methods I would suggest.

- Use a more advanced optimizer (e.g. Adam)
- Add a neuron dropout functionality
- Implement minibatching (splitting training data into smaller batches)
- Experiment with the network architecture
- Implement dynamic learning rate
- Use GPU for the training (e.g. CUDA toolkit for NVIDIA)
- Parallel computing with multiple CPU cores

### Setup Instructions

To run this software, ensure you have the following dependencies:

- **g++ compiler** with **C++11 support**
- **make** (for building the project)
- **Git LFS** (for pulling the MNIST dataset files, as the loader doesn't support compressed format)

Additionally, if you don't want to use Git LFS, you can manually download the dataset from the official [MNIST website](https://yann.lecun.com/exdb/mnist/), uncompress it, and add the files to the `/include/mnist/datasets` directory.

**Clone the repository:**

```bash
git clone https://github.com/pirinenao/cpp_neural_network.git
```

**Change directory && pull the LFS files:**

```bash
cd cpp_neural_network && git lfs pull
```

**Run the Makefile:**

```bash
make
```

**Run the Software:**
```bash
./main
```

### CLI flags

**Available Options:**
| Flag  | Argument        | Argument type             | Description                 | Default  |
| :---: | :---:           | :---:                     | :---:                       | :---:    |
| -e    | epochs          | positive integer value    | set custom number of epochs | 10       |
| -l    | learning rate   | positive float value      | set custom learning rate    | 0.001    | 
| -p    | no arguments    | no arguments              | enable parallel computing   | disabled |
| -h    | no arguments    | no arguments              | print help                  | no value |


### License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for the full text.
