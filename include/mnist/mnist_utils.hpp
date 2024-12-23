//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Contains utility functions to manipulate the MNIST dataset
 */

#ifndef MNIST_UTILS_HPP
#define MNIST_UTILS_HPP

#include <cmath>

namespace mnist
{
    /*!
     * \brief Return the mean value of the elements inside the given range
     * \param container The range to compute the average from
     * \return The average value of the range
     */
    template <typename Container>
    double mean(const Container &container)
    {
        double mean = 0.0;
        for (auto &value : container)
        {
            mean += value;
        }
        return mean / container.size();
    }

    /*!
     * \brief Return the standard deviation of the elements inside the given range
     * \param container The range to compute the standard deviation from
     * \param mean The mean of the given range
     * \return The standard deviation of the range
     */
    template <typename Container>
    double stddev(const Container &container, double mean)
    {
        double std = 0.0;
        for (auto &value : container)
        {
            std += (value - mean) * (value - mean);
        }
        return std::sqrt(std / container.size());
    }

    /*!
     * \brief Normalize the pixel values in the dataset from 0-255 to 0.0-1.0
     * \param dataset The dataset to normalize
     */
    template <typename Dataset>
    void normalize_pixels(Dataset &dataset)
    {
        for (auto &vec : dataset.training_images)
        {
            for (auto &v : vec)
            {
                v = v / 255.0f; // Normalize pixel value to 0.0-1.0
            }
        }

        for (auto &vec : dataset.test_images)
        {
            for (auto &v : vec)
            {
                v = v / 255.0f; // Normalize pixel value to 0.0-1.0
            }
        }
    }

} // end of namespace mnist

#endif
