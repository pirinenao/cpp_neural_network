#ifndef ANN_UTILS_HPP
#define ANN_UTILS_HPP
#include <cmath>
#include <random>

struct LAYER
{
    std::vector<std::vector<float>> weights;
    std::vector<float> biases;
    std::vector<float> weighted_sums;
    std::vector<float> outputs;
    std::vector<float> deltas;
};

struct LOSS
{
    float total_loss;
    float average_loss;
    float current_loss;
};

/*
 * initializes loss vectors to zero
 */
LOSS initialize_loss()
{
    LOSS loss;
    loss.total_loss = 0.0f;
    loss.average_loss = 0.0f;
    loss.current_loss = 0.0f;
    return loss;
}

/*
 * initialize layer weights and biases
 */
LAYER initialize_layer(int inputs, int neurons)
{
    LAYER layer;

    // use random device and normal distribution
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, std::sqrt(2.0f / inputs));

    // initialize weights with He initialization
    layer.weights = std::vector<std::vector<float>>(neurons, std::vector<float>(inputs));
    for (int i = 0; i < neurons; ++i)
    {
        for (int j = 0; j < inputs; ++j)
        {
            layer.weights[i][j] = dist(gen);
        }
    }

    // initialize vectors with zeros
    layer.biases = std::vector<float>(neurons, 0.0f);
    layer.weighted_sums = std::vector<float>(neurons, 0.0f);
    layer.outputs = std::vector<float>(neurons, 0.0f);
    layer.deltas = std::vector<float>(neurons, 0.0f);

    return layer;
}

/*
 * ReLU activation for the hidden layer
 */
float relu(float x)
{
    return fmax(0, x);
}

/*
 * computes the weighted sums for the neurons in the layer
 */
void forward_feed(LAYER *layer,
                  const mnist::MNIST_dataset<std::vector, std::vector<float>, float> &dataset,
                  int sample_index, int neurons)
{
    // reset weighted sums for this forward pass
    std::fill(layer->weighted_sums.begin(), layer->weighted_sums.end(), 0.0f);

    for (int i = 0; i < neurons; i++)
    {
        // calculate weighted sum
        for (size_t j = 0; j < layer->weights[i].size(); j++)
        {
            layer->weighted_sums[i] += (layer->weights[i][j] * dataset.training_images[sample_index][j]);
        }

        // add bias and apply activation function (ReLU)
        layer->outputs[i] = relu(layer->weighted_sums[i] + layer->biases[i]);
    }
}

/*
 * computes the weighted sums for the neurons in the output layer
 */
void feed_output(LAYER *layer, LAYER *input_layer, int neurons)
{
    // initialize weighted sums vector
    std::fill(layer->weighted_sums.begin(), layer->weighted_sums.end(), 0.0f);

    for (int i = 0; i < neurons; i++)
    {
        // calculate weighted sum
        for (size_t j = 0; j < layer->weights[i].size(); j++)
        {
            layer->weighted_sums[i] += layer->weights[i][j] * input_layer->outputs[j];
        }

        // add bias
        // no activation function needed here for output layer, softmax will be applied outside
        layer->outputs[i] = layer->weighted_sums[i] + layer->biases[i];
    }
}

/*
 * softmax activation for the output layers
 * returns a vector of probabilities which sums to 1
 */
std::vector<float> softmax(LAYER *layer, int num_classes)
{
    // initialize output vector
    std::vector<float> output(num_classes, 0.0f);

    // find the maximum value in the outputs for numerical stability
    float max_value = *std::max_element(layer->outputs.begin(), layer->outputs.end());

    // compute exponentiated values and accumulate their sum
    float sum_exp = 0.0;
    for (int i = 0; i < num_classes; ++i)
    {
        // subtract max for stability
        float exp_value = std::exp(layer->outputs[i] - max_value);
        // store the exponentiated value in the output
        output[i] = exp_value;
        // accumulate the sum of exponentiated values
        sum_exp += exp_value;
    }

    // normalize the exponentiated values (to get probabilities)
    for (size_t i = 0; i < output.size(); ++i)
    {
        // normalize each value by the sum of exponentiated values
        output[i] /= sum_exp;
    }

    return output;
}

/*
 * computes the cross-entropy loss
 * sparse means that the labels are integers not one-hot encoded
 */
float sparse_cross_entropy_loss(std::vector<float> &softmax_output, int true_label)
{
    // get the predicted probability for the true class
    float predicted_probability = softmax_output[true_label];

    // add a small epsilon to prevent log(0)
    const float epsilon = 1e-10;
    predicted_probability = std::max(predicted_probability, epsilon); // Ensure it's not zero

    // compute the negative log of the predicted probability
    return -std::log(predicted_probability);
}

/*
 * backpropagate the output layer
 * update the weights and biases based on the error
 */
void backpropagate_output(LAYER &layer, LAYER &input_layer, int expected_class, float learning_rate)
{
    std::vector<float> output_errors(layer.outputs.size());
    std::vector<float> output_deltas(layer.outputs.size());

    // initialize errors to 0
    std::fill(output_errors.begin(), output_errors.end(), 0.0f);

    // for softmax + cross-entropy, the error is the difference between predicted and target
    output_errors[expected_class] = 1.0f - layer.outputs[expected_class];

    // calculate deltas (difference between predicted output and target)
    for (size_t i = 0; i < layer.outputs.size(); i++)
    {
        output_deltas[i] = layer.outputs[i] - (i == (size_t)expected_class ? 1.0f : 0.0f);
    }

    // update weights and biases for the output layer
    for (size_t i = 0; i < layer.outputs.size(); i++) // Loop over output neurons
    {
        for (size_t j = 0; j < input_layer.outputs.size(); j++) // Loop over input neurons
        {
            // update the weight based on gradient descent
            layer.weights[i][j] -= learning_rate * output_deltas[i] * input_layer.outputs[j];
        }

        // update the bias for the output neuron
        layer.biases[i] -= learning_rate * output_deltas[i];
        layer.deltas[i] = output_deltas[i];
    }
}

/*
 * backpropagate the hidden layer
 * update the weights and biases based on the error
 */
void backpropagate_hidden(LAYER &layer, LAYER &next_layer,
                          const mnist::MNIST_dataset<std::vector, std::vector<float>, float> &dataset, int sample_index, float learning_rate)
{
    std::vector<float> hidden_errors(layer.outputs.size());
    std::vector<float> hidden_deltas(layer.outputs.size());

    // initialize errors vector to 0
    std::fill(hidden_errors.begin(), hidden_errors.end(), 0.0f);

    // compute error for the hidden layer neurons
    for (size_t i = 0; i < layer.outputs.size(); i++)
    {
        hidden_errors[i] = 0.0f;
        // sum the errors weighted by the next layer's weights
        for (size_t j = 0; j < next_layer.deltas.size(); j++)
        {
            hidden_errors[i] += next_layer.deltas[j] * next_layer.weights[j][i];
        }

        // calculate the delta for the layer
        hidden_deltas[i] = hidden_errors[i] * (layer.outputs[i] > 0 ? 1.0f : 0.0f);
    }

    // update weights and biases for the layer
    for (size_t i = 0; i < layer.weights.size(); i++)
    {
        for (size_t j = 0; j < layer.weights[i].size(); j++)
        {
            // update weight based on gradient descent
            layer.weights[i][j] -= learning_rate * hidden_deltas[i] * dataset.training_images[sample_index][j];
        }

        // update biases (one bias per neuron in the hidden layer)
        layer.biases[i] -= learning_rate * hidden_deltas[i];
    }
}

#endif