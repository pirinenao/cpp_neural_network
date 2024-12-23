#ifndef PROGRESS_BAR_HPP
#define PROGRESS_BAR_HPP

#include <iostream>
#include <chrono>
#include <thread>

void progress_bar(int progress, int total, int epoch)
{
    // Calculate the percentage of completion
    float percent = (float)progress / total * 100;
    int barWidth = 40;

    std::cout << "[";
    int pos = barWidth * progress / total;
    for (int i = 0; i < barWidth; ++i)
    {
        if (i < pos)
            std::cout << "#";
        else
            std::cout << " ";
    }
    std::cout << "] " << int(percent) << " %\r";
    std::cout << "[epoch " << epoch + 1 << "]";

    std::cout.flush();
}

#endif