#ifndef EVALUATION_HPP
#define EVALUATION_HPP
#include <vector>

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
float sparse_cross_entropy_loss(std::vector<float> &softmax_output, int true_label);

/*
 * find the index of the maximum value in a vector
 * used to find the predicted class
 */
int max_value_index(std::vector<float> &vector);

#endif