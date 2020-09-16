#include "ExtractContour.h"
#include "MapsData.h"
#include "CMLHelper.h"
using namespace Monster;

string ExtractContour::mVisualFolder = "";

bool ExtractContour::extract(vector<vector<double>> &sketch, vector<vec2d> &contour, string visualFolder)
{
    int imgHeight = (int)sketch.size();
    int imgWidth = (int)sketch[0].size();

    mVisualFolder = visualFolder;

    // Mark bool flag and find start point
    vector<vector<bool>> boolSketch(imgHeight, vector<bool>(imgWidth, false));
    vector<vector<bool>> inspected(imgHeight, vector<bool>(imgWidth, false));
    vec2i startPoint(-1, -1);
    int foreground = 0;
    for (int row = 0; row < imgHeight; row++)
    {
        for (int col = 0; col < imgWidth; col++)
        {
            boolSketch[row][col] = (sketch[row][col] > 0.0);

            if (boolSketch[row][col] && startPoint[0] < 0)
            {
                startPoint = vec2i(row, col);
                inspected[row][col] = true;
            }
        }
    }

    vector<vec2i> contourPixels;
    extractOutline(boolSketch, contourPixels);

    return true;
}

bool ExtractContour::extractOutline(vector<vector<bool>> &sketch, vector<vec2i> &outline)
{
    int imgHeight = (int)sketch.size();
    int imgWidth = (int)sketch[0].size();

    vector<vec2i> offset4 = {vec2i(1, 0), vec2i(0, 1), vec2i(-1, 0), vec2i(0, -1)};
    vector<vec2i> offset8 = {vec2i(1, 1), vec2i(1, 0), vec2i(0, 1), vec2i(1, -1),
                             vec2i(-1, 1), vec2i(-1, 0), vec2i(0, -1), vec2i(-1, -1)};

    // close operation
    vector<vector<bool>> mask = sketch;
    int numOperations = imgHeight / 128;
    for (int k = 0; k < numOperations; k++)
    {
        if (!dilateMask(mask))
            return false;
    }

    for (int k = 0; k < numOperations; k++)
    {
        if (!erodeMask(mask))
            return false;
    }

    // Mark background
    vector<vector<int>> labels(imgHeight, vector<int>(imgWidth, -1));
    if (true)
    {
        labels[0][0] = 0;
        vector<vec2i> queue(1, vec2i(0, 0));
        int head = 0;
        while (head < (int)queue.size())
        {
            vec2i curPos = queue[head];
            for (vec2i offset : offset4)
            {
                vec2i nextPos = curPos + offset;
                if (nextPos[0] < 0 || nextPos[0] >= imgHeight || nextPos[1] < 0 || nextPos[1] >= imgWidth)
                    continue;
                if (mask[nextPos[0]][nextPos[1]])
                    continue;
                if (labels[nextPos[0]][nextPos[1]] == 0)
                    continue;
                queue.push_back(nextPos);
                labels[nextPos[0]][nextPos[1]] = 0;
            }
            head++;
        }
    }

    // Mark components
    int mainComponent = -1;
    int mainComponentSize = 0;
    if (true)
    {
        int numComponents = 0;
        for (int row = 0; row < imgHeight; row++)
        {
            for (int col = 0; col < imgWidth; col++)
            {
                if (labels[row][col] >= 0)
                    continue;
                numComponents++;

                labels[row][col] = numComponents;
                vector<vec2i> queue(1, vec2i(row, col));
                int head = 0;

                while (head < (int)queue.size())
                {
                    vec2i curPos = queue[head];
                    for (vec2i offset : offset4)
                    {
                        vec2i nextPos = curPos + offset;
                        if (nextPos[0] < 0 || nextPos[0] >= imgHeight || nextPos[1] < 0 || nextPos[1] >= imgWidth)
                        {
                            continue;
                        }

                        if (labels[nextPos[0]][nextPos[1]] >= 0)
                            continue;

                        queue.push_back(nextPos);
                        labels[nextPos[0]][nextPos[1]] = numComponents;
                    }
                    head++;
                }

                if ((int)queue.size() > mainComponentSize)
                {
                    mainComponentSize = (int)queue.size();
                    mainComponent = numComponents;
                }
            }
        }
    }

    cout << "mainComponent: " << mainComponent << endl;

    // Mark Contour
    MatI flags(imgHeight, vector<int>(imgWidth, 0));
    vec2i startPoint(-1, -1);
    for (int row = 0; row < imgHeight; row++)
    {
        for (int col = 0; col < imgWidth; col++)
        {
            if (labels[row][col] != mainComponent)
                continue;

            bool valid = false;
            for (vec2i offset : offset4)
            {
                int nbRow = cml::clamp(offset[0] + row, 0, imgHeight - 1);
                int nbCol = cml::clamp(offset[1] + col, 0, imgWidth - 1);

                if ((nbRow == row && nbCol == col) || labels[nbRow][nbCol] == 0)
                {
                    valid = true;
                    break;
                }
            }

            if (valid)
            {
                flags[row][col] = true;
                if (startPoint[0] < 0)
                    startPoint = vec2i(row, col);
            }
        }
    }

    cout << (mVisualFolder + "points.png") << endl;
    MapsData::visualizeMask(mVisualFolder + "points.png", flags);

    vector<vec2i> chain(0);
    if (true)
    {
        vector<vec2i> queue(1, startPoint);
        flags[startPoint[0]][startPoint[1]] = false;

        while (!queue.empty())
        {
            vec2i curPos = queue.back();
            bool hasNext = false;

            for (vec2i offset : offset8)
            {
                vec2i nextPos = curPos + offset;
                if (nextPos[0] < 0 || nextPos[0] >= imgHeight || nextPos[1] < 0 || nextPos[1] >= imgWidth)
                    continue;

                if (!flags[nextPos[0]][nextPos[1]])
                    continue;

                flags[nextPos[0]][nextPos[1]] = false;
                queue.push_back(nextPos);
                hasNext = true;
                break;
            }

            if (!hasNext)
            {
                if (queue.size() > chain.size())
                {
                    chain = queue;
                }
                queue.pop_back();
            }
        }
    }

    cout << "Chain Length = " << chain.size() << endl;
    cout << "Area = " << mainComponentSize << endl;
    cout << "Ratio = " << (cml::sqr(double(chain.size())) / mainComponentSize) << endl;

    outline.clear();
    for (vec2i point : chain)
    {
        if (sketch[point[0]][point[1]])
            outline.push_back(point);
    }

    return true;
}

bool ExtractContour::dilateMask(vector<vector<bool>> &mask)
{

    int height = (int)mask.size();
    int width = (int)mask[0].size();

    vector<vector<bool>> result = mask;
    for (int row = 0; row < height; row++)
    {
        for (int col = 0; col < width; col++)
        {
            if (mask[row][col])
            {
                if (row > 0)
                    result[row - 1][col] = true;
                if (row + 1 < height)
                    result[row + 1][col] = true;
                if (col > 0)
                    result[row][col - 1] = true;
                if (col + 1 < width)
                    result[row][col + 1] = true;
                if (row > 0 && col > 0)
                    result[row - 1][col - 1] = true;
                if (row > 0 && col + 1 < width)
                    result[row - 1][col + 1] = true;
                if (row + 1 < height && col > 0)
                    result[row + 1][col - 1] = true;
                if (row + 1 < height && col + 1 < width)
                    result[row + 1][col + 1] = true;
            }
        }
    }
    mask.swap(result);

    return true;
}

bool ExtractContour::erodeMask(vector<vector<bool>> &mask)
{

    int height = (int)mask.size();
    int width = (int)mask[0].size();

    vector<vector<bool>> result = mask;
    for (int row = 0; row < height; row++)
    {
        for (int col = 0; col < width; col++)
        {
            if (mask[row][col])
            {
                bool valid = true;
                if (row > 0 && !mask[row - 1][col])
                    valid = false;
                if (row + 1 < height && !mask[row + 1][col])
                    valid = false;
                if (col > 0 && !mask[row][col - 1])
                    valid = false;
                if (col + 1 < width && !mask[row][col + 1])
                    valid = false;
                result[row][col] = valid;
            }
        }
    }
    mask.swap(result);

    return true;
}