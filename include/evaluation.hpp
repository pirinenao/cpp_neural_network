#ifndef EVALUATION_HPP
#define EVALUATION_HPP
#include <vector>
#include <iostream>
#include <iomanip>

/*
 * Evaluation struct to store evaluation metrics
 */
struct EVALUATION
{
    // loss variables
    float total_loss;
    float average_loss;
    float current_loss;
    // classification variables
    std::vector<int> predictions;
    std::vector<int> true_labels;
    std::vector<std::vector<int>> confusion_matrix;
    // timer variables
    std::chrono::high_resolution_clock::time_point start_time;
    std::chrono::high_resolution_clock::time_point end_time;
    std::chrono::duration<double, std::milli> elapsed;

    /*
     *  initialize loss variables
     */
    void initialize_loss()
    {
        this->total_loss = 0.0f;
        this->average_loss = 0.0f;
        this->current_loss = 0.0f;
    }

    /*
     *  set the predicted labels and true labels
     */
    void set_labels(std::vector<int> predictions, std::vector<int> true_labels)
    {
        this->predictions = predictions;
        this->true_labels = true_labels;
    }

    /*
     *  set the loss for the current sample
     */
    void set_loss(float loss, int sample_index)
    {
        this->current_loss = loss;
        this->total_loss += loss;
        this->average_loss = (this->average_loss * sample_index + this->current_loss) / (sample_index + 1);
    }

    /*
     *  calculate the accuracy of the model
     */
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

    /*
     *  display the confusion matrix
     *  which shows the number of correct and incorrect predictions
     */
    void display_confusion_matrix(int num_classes)
    {
        // check if predictions and true labels are the same size
        if (predictions.size() != true_labels.size())
        {
            std::cout << "Error: predictions and true labels are not the same size\n";
            return;
        }

        // initialize confusion matrix
        this->confusion_matrix = std::vector<std::vector<int>>(num_classes, std::vector<int>(num_classes, 0));

        // fill confusion matrix
        for (size_t i = 0; i < predictions.size(); ++i)
        {
            this->confusion_matrix[predictions[i]][true_labels[i]]++;
        }

        // print confusion matrix
        std::cout << std::endl
                  << "Confusion matrix:\n";
        for (size_t i = 0; i < confusion_matrix.size(); ++i)
        {
            std::cout << "[";
            for (size_t j = 0; j < confusion_matrix[i].size(); ++j)
            {
                if (j > 0)
                    std::cout << ",";

                std::cout << std::setw(5) << confusion_matrix[i][j];
            }
            std::cout << "]" << std::endl;
        }
    }

    /*
     *  calculate precision for each class
     */
    void display_precision(int num_classes)
    {
        // ensure confusion matrix is populated
        if (confusion_matrix.empty())
        {
            std::cout << "Error: confusion matrix is empty. Please compute it first.\n";
            return;
        }

        std::cout << std::endl
                  << "Precisions " << std::endl;

        // calculate precision for each class
        for (int i = 0; i < num_classes; ++i)
        {
            int true_positives = confusion_matrix[i][i];
            int false_positives = 0;

            // sum the column values (false positives for class i)
            for (int j = 0; j < num_classes; ++j)
            {
                if (i != j)
                {
                    false_positives += confusion_matrix[j][i];
                }
            }

            // precision formula: TP / (TP + FP)
            double precision = 0.0;
            if (true_positives + false_positives > 0)
            {
                precision = static_cast<double>(true_positives) / (true_positives + false_positives);
            }

            // print precision for class i with 2 decimal places
            std::cout << "class " << i << ": "
                      << std::fixed << std::setprecision(2) << precision << std::endl;
        }
    }

    /*
     * start timer
     */
    void start_timer()
    {
        this->start_time = std::chrono::high_resolution_clock::now();
    }

    /*
     * end timer
     */
    void end_timer()
    {
        this->end_time = std::chrono::high_resolution_clock::now();
        this->elapsed = this->end_time - this->start_time;
    }

    /*
     * print training metrics
     */
    void print_training_metrics()
    {
        std::cout << std::endl
                  << "avg.loss: " << this->average_loss
                  << " time: " << (int)this->elapsed.count() << " ms\n"
                  << std::endl;
    }

    /*
     * print evaluation metrics
     */
    void print_metrics()
    {
        std::cout << std::endl
                  << std::endl
                  << "avg.loss: " << this->average_loss << std::endl
                  << "accuracy: " << this->accuracy() * 100 << "%"
                  << std::endl;
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