#include "OptimizeDepths.h"
#include "MeshView.h"
#include <iostream>
#include <float.h>

using namespace Monster;
using namespace std;

bool OptimizeDepths::optimize(
    const vector<MatI> &masks,
    const vector<MatD> &depths,
    const vector<Mat3D> &normals,
    const MatD &maskProbs,
    MatI &outMasks,
    MatD &outDepths,
    Mat3D &outNormals)
{
    const double lambda = 0.3; // param: lambda parameter for energy term
    const double alpha = 3.0;  //param: scale factor for original view

    int numViews = (int)masks.size();
    int imgSize = (int)masks[0].size();

    // build valid pixel index
    vector<vec2i> pixelIndex(0);
    MatI pixelMap(imgSize, vector<int>(imgSize, -1));
    pixelIndex.reserve(imgSize * imgSize);
    for (int row = 0; row < imgSize; row++)
    {
        for (int col = 0; col < imgSize; col++)
        {
            if (masks[0][row][col])
            {
                vec2i index(row, col);
                pixelMap[row][col] = (int)pixelIndex.size();
                pixelIndex.push_back(index);
            }
        }
    }
    int numPixels = (int)pixelIndex.size();

    // find neighbors for diescrete partial derivatives, for each pixel, for corresponding 14 views, find valid pixel
    // #pixels, #views, #valid pixels
    vector<vector<vector<vec2i>>> pixelNeighborsDZDx(numPixels, vector<vector<vec2i>>(numViews, vector<vec2i>(0)));
    vector<vector<vector<vec2i>>> pixelNeighborsDZDy(numPixels, vector<vector<vec2i>>(numViews, vector<vec2i>(0)));
    vector<vec2i> neighborOffsets(0);
    for (int ro = -1; ro <= 1; ro++)
    {
        for (int co = -1; co <= 1; co++)
        {
            neighborOffsets.push_back(vec2i(ro, co));
        }
    }
    //    Note: depth direction is opposite to Z axis, dz should be negated
    vector<int> neighborWeightsDZDx = {1, 0, -1, 4, 0, -4, 1, 0, -1};
    vector<int> neighborWeightsDZDy = {-1, -4, -1, 0, 0, 0, 1, 4, 1};

    // #pragma omp parallel for
    for (int pixelID = 0; pixelID < numPixels; pixelID++)
    {
        // p in foreground view
        int row = pixelIndex[pixelID][0];
        int col = pixelIndex[pixelID][1];

        for (int viewID = 0; viewID < numViews; viewID++)
        {
            vector<vec2i> neighborsX(0);
            vector<vec2i> neighborsY(0);
            vector<double> neighborGapsX(0);
            vector<double> neighborGapsY(0);

            double minGapX = DBL_MAX;
            double minGapY = DBL_MAX;

            for (int offsetID = 0; offsetID < (int)neighborOffsets.size(); offsetID++)
            {
                // projected p (denoted as p') in other view (v')
                int nbRow = row + neighborOffsets[offsetID][0];
                int nbCol = col + neighborOffsets[offsetID][1];

                if (nbRow < 0 || nbRow >= imgSize)
                    continue;
                if (nbCol < 0 || nbCol >= imgSize)
                    continue;

                // if p' is marked as background, exclude from energy minimization
                if (!masks[viewID][nbRow][nbCol])
                    continue;

                int neighborID = pixelMap[nbRow][nbCol];
                if (neighborID < 0)
                    continue;
                double gap = fabs(depths[viewID][nbRow][nbCol] - depths[viewID][row][col]); // tagent p',v'
                if (neighborWeightsDZDx[offsetID])
                {
                    neighborsX.push_back(vec2i(neighborID, neighborWeightsDZDx[offsetID]));
                    neighborGapsX.push_back(gap);
                    minGapX = min(minGapX, gap);
                }
                if (neighborWeightsDZDy[offsetID])
                {
                    neighborsY.push_back(vec2i(neighborID, neighborWeightsDZDy[offsetID]));
                    neighborGapsY.push_back(gap);
                    minGapY = min(minGapY, gap);
                }
            }

            double maxGapX = minGapX * 10.0; // param: depth discontinuity threshold
            double maxGapY = minGapY * 10.0;
            int neighborWeightsX = 0;
            int neighborWeightsY = 0;
            for (int id = 0; id < (int)neighborsX.size(); id++)
            {
                if (neighborGapsX[id] < maxGapX)
                {
                    pixelNeighborsDZDx[pixelID][viewID].push_back(neighborsX[id]); // record neighbor pixel and corresponding weight
                    neighborWeightsX += neighborsX[id][1];
                }
            }

            if (neighborWeightsX) // TODO: Why
            {
                pixelNeighborsDZDx[pixelID][viewID].push_back(vec2i(pixelID, -neighborWeightsX));
            }

            for (int id = 0; id < (int)neighborsY.size(); id++)
            {
                if (neighborGapsY[id] < maxGapY)
                {
                    pixelNeighborsDZDy[pixelID][viewID].push_back(neighborsY[id]);
                    neighborWeightsY += neighborsY[id][1];
                }
            }

            if (neighborWeightsY)
            {
                pixelNeighborsDZDy[pixelID][viewID].push_back(vec2i(pixelID, -neighborWeightsY));
            }
        }
    }

    if (false)
    {
        int countX = 0, countY = 0;
        int countTX = 0, countTY = 0;
        for (int pixelID = 0; pixelID < numPixels; pixelID++)
        {
            for (int viewID = 0; viewID < numViews; viewID++)
            {
                int x = pixelNeighborsDZDx[pixelID][viewID].size();
                int y = pixelNeighborsDZDy[pixelID][viewID].size();
                if (x)
                {
                    countX += x;
                    countTX++;
                }
                if (y)
                {
                    countY += y;
                    countTY++;
                }
            }
        }
        // Each foreground pixel often has 3-4 corresponding pixels in other views
        cout << "avg neighbors: " << (countX / (double)countTX) << ", " << (countY / (double)countTY) << endl;
    }

    // build sparse linear system
    double kappa = MeshView::VIEW_RADIUS * 2 / imgSize;
    vector<Eigen::Triplet<double, int>> triplets(0);
    vector<double> values(0);
    int eqnID = 0;
    for (int pixelID = 0; pixelID < numPixels; pixelID++)
    {
        // p in foreground view
        int row = pixelIndex[pixelID][0];
        int col = pixelIndex[pixelID][1];
        double probWeight = maskProbs[row][col];

        for (int viewID = 0; viewID < numViews; viewID++)
        {
            if (!masks[viewID][row][col])
                continue;
            double Zuv = depths[viewID][row][col];
            auto Nuv = normals[viewID][row][col];
            double eqnWeight = (viewID == 0) ? alpha : 1.0;
            eqnWeight *= probWeight;

            // lambda * Z = lambda * Zuv
            if (true)
            {
                double lhs = lambda * eqnWeight;
                double rhs = lambda * Zuv * eqnWeight;
                triplets.push_back(Eigen::Triplet<double, int>(eqnID, pixelID, lhs));
                values.push_back(rhs);
                eqnID++;
            }

            // (1-lambda) * (dZdx * Nuv.z) = (1-lambda) * (-kappa * Nuv.x)
            if (true)
            {
                int totalWeight = 0;
                for (vec2i neighbor : pixelNeighborsDZDx[pixelID][viewID])
                {
                    if (neighbor[0] != pixelID)
                        totalWeight += abs(neighbor[1]);
                }
                if (totalWeight)
                {
                    for (vec2i neighbor : pixelNeighborsDZDx[pixelID][viewID])
                    {
                        int neighborID = neighbor[0];
                        double lhs = (1 - lambda) * Nuv[2] * (neighbor[1] / (double)totalWeight) * eqnWeight;
                        triplets.push_back(Eigen::Triplet<double, int>(eqnID, neighborID, lhs));
                    }
                    double rhs = (1 - lambda) * (-kappa * Nuv[0]) * eqnWeight;
                    values.push_back(rhs);
                    eqnID++;
                }
            }

            // (1-lambda) * (dZdy * Nuv.z) = (1-lcout << "# eqs=" << numEquations << endl;
            // cout << "# pixels=" << numPixels << endl;
        }
    }

    int numEquations = eqnID;
    cout << "# eqs=" << numEquations << endl;
    cout << "# pixels=" << numPixels << endl;

    // solve sparse linear system
    Eigen::SparseMatrix<double> matA(numEquations, numPixels);
    matA.setFromTriplets(triplets.begin(), triplets.end());
    Eigen::VectorXd matB = Eigen::Map<Eigen::VectorXd>(values.data(), values.size());

    // get intial guess
    Eigen::VectorXd matG(numPixels);
    for (int pixelID = 0; pixelID < numPixels; pixelID++)
    {
        int row = pixelIndex[pixelID][0];
        int col = pixelIndex[pixelID][1];

        matG[pixelID] = depths[0][row][col];
    }

    Eigen::LeastSquaresConjugateGradient<Eigen::SparseMatrix<double>> solver;
    solver.compute(matA);
    if (solver.info() != Eigen::Success)
    {
        cerr << "fail to compuate matA" << endl;
        return false;
    }

    Eigen::VectorXd matX = solver.solveWithGuess(matB, matG);
    if (solver.info() != Eigen::Success)
    {
        cerr << "solve error - " + to_string(solver.info());
        return false;
    }

    //output results
    outMasks.assign(imgSize, vector<int>(imgSize, false));
    outDepths.assign(imgSize, vector<double>(imgSize, 1.0));
    outNormals.assign(imgSize, vector<vec3d>(imgSize, vec3d(1.0, 1.0, 1.0)));

    for (int pixelID = 0; pixelID < numPixels; pixelID++)
    {
        int row = pixelIndex[pixelID][0];
        int col = pixelIndex[pixelID][1];
        outMasks[row][col] = true;
        outDepths[row][col] = matX[pixelID];
        if (masks[0][row][col])
        {
            outNormals[row][col] = normals[0][row][col];
        }
        else
        {
            vec3d accumNormal(0.0, 0.0, 0.0);
            for (int viewID = 0; viewID < numViews; viewID++)
            {
                if (masks[viewID][row][col])
                {
                    accumNormal += normals[viewID][row][col];
                }
            }
            if (accumNormal.length_squared())
                accumNormal.normalize();
            else
                accumNormal = vec3d(0.0, 0.0, 1.0);
            outNormals[row][col] = accumNormal;
        }
    }
    return true;
}