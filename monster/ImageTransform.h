#pragma once

#include <iostream>
#include <vector>

using namespace std;

namespace Monster
{
    namespace ImageTransform
    {
        template <class T>
        inline bool flipVertical(const vector<T> &inImg, vector<T> &outImg, int width, int height, int channel)
        {
            if (width * height * channel != (int)inImg.size())
            {
                cerr << "Error: incorrect image size" << endl;
                return false;
            }

            vector<T> buffer(inImg.size());
            int lineSize = width * channel;
            for (int h = 0; h < height; h++)
            {
                auto first = inImg.begin() + h * lineSize;
                auto last = first + lineSize;
                auto dest = buffer.begin() + (height - h - 1) * lineSize;
                copy(first, last, dest);
            }
            outImg.swap(buffer);
            return true;
        }

    } // namespace ImageTransform
} // namespace Monster