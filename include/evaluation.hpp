#ifndef EVALUATION_HPP
#define EVALUATION_HPP

/*
 * Evaluation struct to store evaluation metrics
 */
struct EVALUATION
{
    float total_loss;
    float average_loss;
    float current_loss;
    std::vector<int> predictions;
    std::vector<int> true_labels;

    void initialize_loss()
    {
        this->total_loss = 0.0f;
        this->average_loss = 0.0f;
        this->current_loss = 0.0f;
    }

    void set_labels(std::vector<int> predictions, std::vector<int> true_labels)
    {
        this->predictions = predictions;
        this->true_labels = true_labels;
    }

    void set_loss(float loss, int sample_index)
    {
        this->current_loss = loss;
        this->total_loss += loss;
        this->average_loss = (this->average_loss * sample_index + this->current_loss) / (sample_index + 1);
    }

    double accuracy()
    {
        int correct = 0;
        for (size_t i = 0; i < predictions.size(); ++i)
        {
            // check if prediction is correct
            if (predictions[i] == true_labels[i])
            {
                correct++;
            }
        }
        return (double)correct / predictions.size();
    }
};

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
 * find the index of the maximum value in a vector
 * used to find the predicted class
 */
int max_value_index(std::vector<float> &vector)
{
    // Initialize the index and maximum value
    int max_index = -1;
    float max_value = 0.0f; // Start with the smallest possible integer

    // Loop through the vector to find the largest value
    for (size_t i = 0; i < vector.size(); ++i)
    {
        if (vector[i] > max_value)
        {
            max_value = vector[i];
            max_index = i;
        }
    }

    return max_index;
}

#endif