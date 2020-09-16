#include "UNet.h"
#include <iostream>
using namespace std;
using namespace net;

namespace F = torch::nn::functional;
namespace nn = torch::nn;

UpConvImpl::UpConvImpl(int in_channels, int out_channels, int kernel_size, int stride, std::string activation_fn) : activation_fn(activation_fn)
{
    if (Cfg::conv_type == DE_CONV)
    {
        deconv = nn::ConvTranspose2d(nn::ConvTranspose2dOptions(in_channels, out_channels, kernel_size).stride(stride));
        register_module("deconv", deconv);
    }
    else
    {
        conv = torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, 3).padding(1).stride(1));
        bn = nn::BatchNorm2d(nn::BatchNorm2dOptions(out_channels));
        register_module("conv", conv);
        register_module("bn", bn);
    }
}

torch::Tensor UpConvImpl::forward(const torch::Tensor &x)
{
    if (Cfg::conv_type == DE_CONV)
    {
        return deconv->forward(x);
    }

    torch::Tensor up;
    if (Cfg::conv_type == NEAREST)
        up = F::interpolate(x, F::InterpolateFuncOptions().scale_factor(std::vector<double>({2.0, 2.0})).mode(torch::kNearest));
    else if (Cfg::conv_type == BILINEAR)
        up = F::interpolate(x, F::InterpolateFuncOptions().scale_factor(std::vector<double>({2.0, 2.0})).mode(torch::kBilinear));
    up = conv->forward(up);
    up = bn->forward(up);
    if (activation_fn == "relu")
    {
        up = F::relu(up);
    }
    else if (activation_fn == "leaky_relu")
    {
        up = F::leaky_relu(up);
    }
    else if (activation_fn == "tanh")
    {
        up = up.tanh();
    }

    return up;
}

DecoderImpl::DecoderImpl()
{

    d6 = UpConv(512, 512, 2, 2);
    d5 = UpConv(1024, 512, 2, 2);
    d4 = UpConv(1024, 512, 2, 2);
    d3 = UpConv(1024, 256, 2, 2);
    d2 = UpConv(512, 128, 2, 2);
    d1 = UpConv(256, 64, 2, 2);
    out_layer = UpConv(128, Cfg::out_channels, 2, 2, Cfg::activation_fn);

    register_module("d6", d6);
    register_module("d5", d5);
    register_module("d4", d4);
    register_module("d3", d3);
    register_module("d2", d2);
    register_module("d1", d1);
    register_module("out_layer", out_layer);
}

torch::Tensor DecoderImpl::forward(const torch::Tensor &x1,
                                   const torch::Tensor &x2,
                                   const torch::Tensor &x3,
                                   const torch::Tensor &x4,
                                   const torch::Tensor &x5,
                                   const torch::Tensor &x6,
                                   const torch::Tensor &x7)
{
    torch::Tensor o6 = d6->forward(x7);                      // 512
    torch::Tensor o5 = d5->forward(torch::cat({o6, x6}, 1)); //512+512
    torch::Tensor o4 = d4->forward(torch::cat({o5, x5}, 1)); //512 + 512
    torch::Tensor o3 = d3->forward(torch::cat({o4, x4}, 1));
    torch::Tensor o2 = d2->forward(torch::cat({o3, x3}, 1));
    torch::Tensor o1 = d1->forward(torch::cat({o2, x2}, 1));
    torch::Tensor out = out_layer->forward(torch::cat({o1, x1}, 1));
    // cout << "o6 " << o6.sizes() << endl;
    // cout << "o5 " << o5.sizes() << endl;
    // cout << "o4 " << o4.sizes() << endl;
    // cout << "o3 " << o3.sizes() << endl;
    // cout << "o2 " << o2.sizes() << endl;
    // cout << "o1 " << o1.sizes() << endl;
    // cout << "out " << out.sizes() << endl;

    return out;
}

EncoderImpl::EncoderImpl(int64_t in_channels)
{ // Encoder
    e1 = torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, 64, 3).padding(1).stride(2));
    e2 = torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 128, 3).padding(1).stride(2));
    e3 = torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 256, 3).padding(1).stride(2));
    e4 = torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 512, 3).padding(1).stride(2));
    e5 = torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).padding(1).stride(2));
    e6 = torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).padding(1).stride(2));
    e7 = torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).padding(1).stride(2));

    register_module("e1", e1);
    register_module("e2", e2);
    register_module("e3", e3);
    register_module("e4", e4);
    register_module("e5", e5);
    register_module("e6", e6);
    register_module("e7", e7);
}

EncoderOutput EncoderImpl::forward(const torch::Tensor &x)
{
    EncoderOutput out;
    out.x1 = e1->forward(x);
    out.x2 = e2->forward(out.x1);
    out.x3 = e3->forward(out.x2);
    out.x4 = e4->forward(out.x3);
    out.x5 = e5->forward(out.x4);
    out.x6 = e6->forward(out.x5);
    out.x7 = e7->forward(out.x6);
    // x1: [1, 64, 128, 128]
    // x2: [1, 128, 64, 64]
    // x3: [1, 256, 32, 32]
    // x4: [1, 512, 16, 16]
    // x5: [1, 512, 8, 8]
    // x6: [1, 512, 4, 4]
    // x7: [1, 512, 2, 2]
    // features: 2048
    return out;
}

DiscriminatorImpl::DiscriminatorImpl()
{
    encoder = Encoder(Cfg::out_channels);
    register_module("encoder", encoder);

    fc = nn::Linear(nn::LinearOptions(2048, 1));
    register_module("fc", fc);

    sigmoid = nn::Sigmoid();
    register_module("sigmoid", sigmoid);
}

torch::Tensor DiscriminatorImpl::forward(const torch::Tensor &x)
{
    auto num_batches = x.size(0);
    EncoderOutput e = encoder->forward(x);
    torch::Tensor features = e.x7.view({num_batches, -1});

    auto probs = sigmoid->forward(fc->forward(features));
    return probs;
}

UNetImpl::UNetImpl()
{
    // Encoder
    encoder = Encoder(Cfg::in_channels);
    register_module("encoder", encoder);

    // decoder = Decoder();
    // register_module("decoder", decoder);

    for (int i = 0; i < Cfg::num_views; i++)
    {
        Decoder d;
        decoders.push_back(d);
        char name[20];
        sprintf(name, "decoder%d", i);
        register_module(name, decoders[i]);
    }
}

std::tuple<ViewPred, torch::Tensor, torch::Tensor, torch::Tensor> UNetImpl::forward(const torch::Tensor &x)
{
    EncoderOutput e = encoder->forward(x);

    std::vector<torch::Tensor> preds, masks, depths, normals;
    for (int i = 0; i < Cfg::num_views; i++)
    {
        auto out = decoders[i]->forward(e.x1, e.x2, e.x3, e.x4, e.x5, e.x6, e.x7); // N x 5 x H x W
        auto out_arr = torch::split_with_sizes(out, {1, 1, 3}, 1);
        auto mask = out_arr[0];
        auto depth = out_arr[1];
        auto normal = out_arr[2];
        preds.push_back(out);
        masks.push_back(mask);
        depths.push_back(depth);
        normals.push_back(normal);
        // cout << "out: " << out.sizes() << " mask: " << mask.sizes() << " depth: " << depth.sizes() << " normal: " << normal.sizes() << endl;
    }

    auto mask_batch = torch::cat(masks, 1);
    auto depth_batch = torch::cat(depths, 1);
    auto normal_batch = torch::cat(normals, 1); // N x (#view x 3) x H x W
    return {preds, mask_batch, depth_batch, normal_batch};
}