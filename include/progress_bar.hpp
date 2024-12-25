#ifndef PROGRESS_BAR_HPP
#define PROGRESS_BAR_HPP

#include <iostream>
#include <chrono>
#include <thread>

#define NO_EPOCHS 0

/**
 * display a progress bar
 */
void progress_bar(int progress, int total, int epoch);

#endif