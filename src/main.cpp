#include <iostream>
#include "../include/mnist/mnist_reader.hpp"
#include "../include/mnist/mnist_utils.hpp"

using namespace std;

int main(void)
{
    // load the MNIST dataset
    mnist::MNIST_dataset<vector, vector<uint8_t>, uint8_t> dataset =
        mnist::read_dataset<vector, vector, uint8_t, uint8_t>(MNIST_DATA_LOCATION);

    // print out the dataset sizes to ensure its loaded correctly
    cout << "Number of training images = " << dataset.training_images.size() << endl;
    cout << "Number of training labels = " << dataset.training_labels.size() << endl;
    cout << "Number of test images = " << dataset.test_images.size() << endl;
    cout << "Number of test labels = " << dataset.test_labels.size() << endl;
}