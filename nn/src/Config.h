#pragma once
#include <torch/torch.h>
#include <string>
#include <iostream>

// #ifdef WITH_CUDA
const torch::Device device(torch::kCUDA);
// #else
// const torch::Device device(torch::kCPU);
// #endif

namespace net
{
    enum ConvType
    {
        NEAREST = 0,
        BILINEAR,
        DE_CONV
    };

    struct Cfg // Config
    {
        // Data
        static std::string data_dir;
        static std::string img_dir;

        // Network
        static std::string input_views;
        static std::string style_ids;
        static int out_channels;
        static int num_target_views;
        static int num_views;
        static int in_channels;

        static ConvType conv_type; // for deocder
        static std::string activation_fn;
        static bool is_scale_depth_loss;

        // Training Parameter
        static int epoch;
        static int val_epoch;
        static float lr;
        static float weight_decay;
        static int batch_size;
        static bool with_adversarial;
        static bool with_normal;
        static float lambda_p;
        static float lambda_a;
    };
} // namespace net
