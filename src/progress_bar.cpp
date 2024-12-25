#include "../include/progress_bar.hpp"

void progress_bar(int progress, int total, int epoch)
{
    if (epoch)
    {
        std::cout << "[epoch " << epoch << "]";
    }

    // Calculate the percentage of completion
    float percent = (float)progress / total * 100;
    int bar_width = 40;

    std::cout << "[";
    int pos = bar_width * progress / total;
    for (int i = 0; i < bar_width; ++i)
    {
        if (i < pos)
            std::cout << "#";
        else
            std::cout << " ";
    }

    std::cout << "] " << int(percent) << " %\r";
    std::cout.flush();
}