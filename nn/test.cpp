
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <iostream>

#include "ImgUtil.h"

using namespace std;
using namespace net;

void testCvImageToTensor()
{

    // cv::Mat img = cv::Mat::zeros({3, 3}, CV_8UC3);
    // auto &p00 = img.at<cv::Vec3b>(0, 0);
    // p00.val[0] = 125;
    // p00.val[1] = 0;
    // p00.val[2] = 255;
    // auto &p01 = img.at<cv::Vec3b>(0, 1);
    // p01.val[0] = 1;
    // p01.val[1] = 2;
    // p01.val[2] = 3;

    // cout << img << endl;

    // auto imgTensor = ImgUtil::CvImageToTensor(img);
    // cout << imgTensor << endl;

    cv::Mat img = cv::Mat::zeros({3, 3}, CV_8UC1);
    auto &p00 = img.at<cv::Vec3b>(0, 0);
    p00.val[0] = 125;
    auto &p01 = img.at<cv::Vec3b>(0, 1);
    p01.val[0] = 1;

    cout << img << endl;

    auto imgTensor = ImgUtil::CvImageToTensor(img);
    cout << imgTensor << endl;
}

void test_normalize_img()
{
    cv::Mat im8U = cv::imread("data/Character/sketch/00044_Ichigo/sketch-F-0.png", cv::IMREAD_UNCHANGED);
    cv::Mat im16U = cv::imread("data/Character/dnfs/00044_Ichigo/dnfs-256-0.png", cv::IMREAD_UNCHANGED);

    cv::Mat norm1 = ImgUtil::normalize_img(im8U);
    cv::Mat norm2 = ImgUtil::normalize_img(im16U);

    cout << im16U << endl;
}

void test_unnormalize_img()
{
    cv::Mat im8U = cv::imread("data/Character/sketch/00044_Ichigo/sketch-F-0.png", cv::IMREAD_UNCHANGED);
    cv::Mat im16U = cv::imread("data/Character/dnfs/00044_Ichigo/dnfs-256-0.png", cv::IMREAD_UNCHANGED);

    cv::Mat norm1 = ImgUtil::normalize_img(im8U);
    cv::Mat norm2 = ImgUtil::normalize_img(im16U);

    cout << ImgUtil::unnormalize_img(norm1) << endl;
}

void test_extract_boolean_mask()
{
    cv::Mat im8U = cv::imread("data/Character/sketch/00044_Ichigo/sketch-F-0.png", cv::IMREAD_UNCHANGED);
    // cv::Mat im16U = cv::imread("data/Character/dnfs/00044_Ichigo/dnfs-256-0.png", cv::IMREAD_UNCHANGED);
    cv::Mat im16U = cv::imread("data/Character/dnfs/15127_Barton/dnfs-256-0.png", cv::IMREAD_UNCHANGED);

    cv::Mat norm1 = ImgUtil::normalize_img(im8U);
    cv::Mat norm2 = ImgUtil::normalize_img(im16U);
    cv::Mat mask = ImgUtil::extract_boolean_mask(norm2);

    for (int i = 0; i < mask.rows; i++)
    {
        for (int j = 0; j < mask.cols; j++)
        {
            if (mask.at<float>(i, j) < 0 || mask.at<float>(i, j) > 1)
                cout << i << ", " << j << ", " << mask.at<float>(i, j) << endl;
        }
    }

    cv::imshow("orig", im16U);
    cv::imshow("mask", mask * 255);
    cv::waitKey(-1);
}

void test_extract_depth_normal()
{
    cv::Mat im8U = cv::imread("data/Character/sketch/00044_Ichigo/sketch-F-0.png", cv::IMREAD_UNCHANGED);
    cv::Mat im16U = cv::imread("data/Character/dnfs/00044_Ichigo/dnfs-256-0.png", cv::IMREAD_UNCHANGED);

    cv::Mat norm1 = ImgUtil::normalize_img(im8U);
    cv::Mat norm2 = ImgUtil::normalize_img(im16U);
    cv::Mat normal, depth;
    ImgUtil::extract_depth_normal(norm2, normal, depth);
    cv::Mat unnorm = ImgUtil::unnormalize_img(norm2, 65535.0);
    unnorm.convertTo(unnorm, CV_16UC4);
    normal = ImgUtil::unnormalize_img(normal, 65535.0);
    normal.convertTo(normal, CV_16UC3);
    depth = ImgUtil::unnormalize_img(depth, 65535.0);
    depth.convertTo(depth, CV_16UC1);
    cv::Mat bgra[4];
    cv::split(im16U, bgra);

    cv::imshow("orig", im16U);
    cv::imshow("depth", bgra[3]);
    cv::imshow("unnorm", unnorm);
    cv::imshow("normal", normal);
    cv::imshow("depth", depth);
    cv::waitKey(-1);
}

void testTensorToCvMat()
{
    cv::Mat im8U = cv::imread("data/Character/sketch/00044_Ichigo/sketch-F-0.png", cv::IMREAD_UNCHANGED);
    cv::Mat im16U = cv::imread("data/Character/dnfs/00044_Ichigo/dnfs-256-0.png", cv::IMREAD_UNCHANGED);

    cv::Mat norm1 = ImgUtil::normalize_img(im8U);
    cv::Mat norm2 = ImgUtil::normalize_img(im16U);

    cv::Mat normal, depth;
    ImgUtil::extract_depth_normal(norm2, normal, depth);

    auto img1 = ImgUtil::CvImageToTensor(normal);
    auto img2 = ImgUtil::CvImageToTensor(depth);

    cout << "img2: " << img2.sizes() << endl;
    cout << "img2: " << img2.unsqueeze(0).sizes() << endl;

    normal = ImgUtil::TensorToCvMat(img1);
    cout << __FILE__ << " " << __LINE__ << endl;
    depth = ImgUtil::TensorToCvMat(img2.unsqueeze(0));
    cout << __FILE__ << " " << __LINE__ << endl;

    normal = ImgUtil::unnormalize_img(normal, 65535.0);
    normal.convertTo(normal, CV_16UC3);
    cout << __FILE__ << " " << __LINE__ << endl;
    depth = ImgUtil::unnormalize_img(depth, 65535.0);
    depth.convertTo(depth, CV_16UC1);

    cv::imshow("normal", normal);
    cv::imshow("depth", depth);
    cv::waitKey(-1);
}
int main()
{
    // testCvImageToTensor();
    // test_unnormalize_img();
    test_extract_boolean_mask();
    // test_extract_depth_normal();
    // testTensorToCvMat();
    return 0;
}