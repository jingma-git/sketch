#pragma once

#include <torch/torch.h>
#include <vector>
#include "Config.h"

namespace net
{
    class UpConvImpl : public torch::nn::Module
    {
    public:
        UpConvImpl(int in_channels, int out_channels, int kernel_size, int stride, std::string activation_fn = "relu");

        torch::Tensor forward(const torch::Tensor &x);

    private:
        torch::nn::Conv2d conv{nullptr};
        torch::nn::BatchNorm2d bn{nullptr};
        torch::nn::ConvTranspose2d deconv{nullptr};
        std::string activation_fn;
    };
    TORCH_MODULE(UpConv);

    class DecoderImpl : public torch::nn::Module
    {
    public:
        DecoderImpl();

        torch::Tensor forward(const torch::Tensor &x1,
                              const torch::Tensor &x2,
                              const torch::Tensor &x3,
                              const torch::Tensor &x4,
                              const torch::Tensor &x5,
                              const torch::Tensor &x6,
                              const torch::Tensor &x7);

        UpConv d6{nullptr};
        UpConv d5{nullptr};
        UpConv d3{nullptr};
        UpConv d4{nullptr};
        UpConv d2{nullptr};
        UpConv d1{nullptr};
        UpConv out_layer{nullptr};
    };
    TORCH_MODULE(Decoder);

    struct EncoderOutput
    {
        torch::Tensor x1;
        torch::Tensor x2;
        torch::Tensor x3;
        torch::Tensor x4;
        torch::Tensor x5;
        torch::Tensor x6;
        torch::Tensor x7;
    };

    class EncoderImpl : public torch::nn::Module
    {
    public:
        EncoderImpl(int64_t in_channels);

        EncoderOutput forward(const torch::Tensor &);

    private:
        torch::nn::Conv2d e1{nullptr};
        torch::nn::Conv2d e2{nullptr};
        torch::nn::Conv2d e3{nullptr};
        torch::nn::Conv2d e4{nullptr};
        torch::nn::Conv2d e5{nullptr};
        torch::nn::Conv2d e6{nullptr};
        torch::nn::Conv2d e7{nullptr};
    };
    TORCH_MODULE(Encoder);

    // entry in the vector is predicted mask_depth_normal tensor under 14 views
    typedef std::vector<torch::Tensor> ViewPred;
    class DiscriminatorImpl : public torch::nn::Module
    {
    public:
        DiscriminatorImpl();

        torch::Tensor forward(const torch::Tensor &);

    private:
        Encoder encoder{nullptr};
        torch::nn::Linear fc{nullptr};
        torch::nn::Sigmoid sigmoid{nullptr};
    };
    TORCH_MODULE(Discriminator);

    class UNetImpl : public torch::nn::Module
    {
    public:
        UNetImpl();

        // return: ViewPred, pred_mask, pred_depth, pred_normal
        std::tuple<ViewPred, torch::Tensor, torch::Tensor, torch::Tensor> forward(const torch::Tensor &);

    private:
        Encoder encoder{nullptr};
        std::vector<Decoder> decoders;
    };

    TORCH_MODULE(UNet);
} // namespace net
