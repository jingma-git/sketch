
#include <torch/torch.h>
#include <iostream>
#include <opencv2/opencv.hpp>

#include "Config.h"
#include "SketchDataset.h"
#include "UNet.h"
#include "loss.h"
#include "ImgUtil.h"

using namespace std;
using namespace net;
namespace F = torch::nn::functional;

void train()
{
    SketchDataset train_set("validate-list-mini.txt");
    SketchDataset val_set("validate-list-mini.txt");
    auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
        train_set,
        torch::data::DataLoaderOptions().batch_size(Cfg::batch_size).workers(0));
    auto val_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
        val_set,
        torch::data::DataLoaderOptions().batch_size(1).workers(0));

    UNet unet;
    unet->to(device);
    Discriminator discriminator;
    discriminator->to(device);
    torch::optim::Adam optim_G(unet->parameters(),
                               torch::optim::AdamOptions(Cfg::lr).weight_decay(Cfg::weight_decay));
    torch::optim::Adam optim_D(discriminator->parameters(),
                               torch::optim::AdamOptions(Cfg::lr).weight_decay(Cfg::weight_decay));

    int step = 0;
    for (int epoch = 0; epoch < Cfg::epoch; ++epoch)
    {
        unet->train();
        for (auto &input : *train_loader)
        {
            optim_G.zero_grad();

            std::vector<torch::Tensor> images;
            std::vector<torch::Tensor> gt_masks;
            std::vector<torch::Tensor> gt_depths;
            std::vector<torch::Tensor> gt_normals;
            std::vector<torch::Tensor> normal_masks;
            std::vector<ViewPred> gt_views;
            for (size_t i = 0; i < input.size(); i++)
            {
                input[i].data.image = input[i].data.image.to(device);
                input[i].target.gt_masks = input[i].target.gt_masks.to(device);
                input[i].target.gt_depths = input[i].target.gt_depths.to(device);
                input[i].target.gt_normals = input[i].target.gt_normals.to(device);
                input[i].target.normal_masks = input[i].target.normal_masks.to(device);
                input[i].target.targets = input[i].target.targets.to(device); // #views x 5 x H x W

                images.push_back(input[i].data.image);
                gt_masks.push_back(input[i].target.gt_masks); // #views x H x W
                gt_depths.push_back(input[i].target.gt_depths);
                gt_normals.push_back(input[i].target.gt_normals);     // (#views x 3) x H x W
                normal_masks.push_back(input[i].target.normal_masks); // (#views x 3) x H x W
                auto gt_view_vec = torch::split(input[i].target.targets, 1, 0);
                gt_views.push_back(gt_view_vec);
            }
            //assert(gt_views.size() == Cfg::batch_size);

            auto img_batch = torch::stack(images); // N x 2 x H x W for input_views = "FS"
            auto gt_mask_batch = torch::stack(gt_masks);
            auto gt_depth_batch = torch::stack(gt_depths);
            auto gt_normal_batch = torch::stack(gt_normals);     // N x (#views x 3) x H x W
            auto normal_mask_batch = torch::stack(normal_masks); // N x (#views x 3) x H x W

            std::vector<torch::Tensor> view_preds;
            torch::Tensor pred_masks, pred_depths, pred_normals;
            std::tie(view_preds, pred_masks, pred_depths, pred_normals) = unet->forward(img_batch); // 2 x 5 x H x W
            if (false)
            {
                // cout << "inputs: " << img_batch.sizes() << endl;
                // cout << "gt_mask: " << gt_mask_batch.sizes() << endl;
                // cout << "normal_mask: " << normal_mask_batch.sizes() << endl;
                // cout << "gt_depth: " << gt_depth_batch.sizes() << endl;
                // cout << "gt_normal: " << gt_normal_batch.sizes() << endl;
                // cout << "pred_mask: " << pred_masks.sizes() << endl;
                // cout << "pred_depth: " << pred_depths.sizes() << endl;
                // cout << "pred_normal: " << pred_normals.sizes() << endl;
            }
            pred_depths = torch::clamp(pred_depths, -1.0, 1.0); //clamp the predicted depths
            auto mask_loss = cal_mask_loss(gt_mask_batch, pred_masks);
            auto depth_loss = cal_depth_loss(gt_mask_batch, gt_depth_batch, pred_depths);
            auto normal_loss = cal_normal_loss(gt_depth_batch, normal_mask_batch, gt_normal_batch, pred_normals);
            // Prediction loss for Generator
            auto loss_p = mask_loss + depth_loss + normal_loss;
            if (Cfg::with_adversarial)
            {
                // Memory layout [fake_input]:
                //              view0-batch 1 (1x5xNxW)
                //                    batch 2 (1x5xNxW)
                //              view1-batch 1 (1x5xNxW)
                //                    batch 2 (1x5xNxW)
                //              .
                //              .
                //              .
                //             view13-batch 1 (1x5xNxW)
                //                    batch 2 (1x5xNxW)
                torch::Tensor fake_input = torch::cat(view_preds, 0);
                std::vector<torch::Tensor> real_input_vec;
                for (int i = 0; i < Cfg::num_views; i++)
                {
                    std::vector<torch::Tensor> tmp;
                    for (int batch_i = 0; batch_i < Cfg::batch_size; batch_i++)
                    {
                        tmp.push_back(gt_views[batch_i][i]); // 1x5xHxW
                    }
                    real_input_vec.push_back(torch::cat(tmp, 0));
                }
                torch::Tensor real_input = torch::cat(real_input_vec, 0);
                auto probs_g = discriminator->forward(fake_input);
                auto probs_real = discriminator->forward(real_input);
                auto probs_fake = discriminator->forward(fake_input.detach());
                if (false)
                {
                    // cout << "fake_input: " << fake_input.sizes() << endl;
                    // cout << "real_input: " << real_input.sizes() << endl;
                    // cout << "discriminator_input" << dis_input.sizes() << endl;
                    // cout << "probs" << probs.sizes() << endl;
                    // cout << "probs_real" << probs_real.sizes() << endl;
                    // cout << "probs_fake" << probs_fake.sizes() << endl;
                }
                // adversarial loss for Generator, the generated Fake Data should maximize the probability
                // which means the generated data should be as real as possible to fool the discriminator
                // auto loss_g_a = torch::sum(-torch::log(torch::clamp_min(probs_g, 1e-6)));
                auto label_g = torch::ones_like(probs_g);
                auto loss_g_a = F::binary_cross_entropy(probs_g, label_g);
                // adversarial loss for Discriminator, discriminator should give the 'real data' high score
                // auto loss_d_r = torch::sum(-torch::log(torch::clamp_min(probs_real, 1e-6)));
                auto label_real = torch::ones_like(probs_real);
                auto loss_d_r = F::binary_cross_entropy(probs_real, label_real);
                // adversarial loss for Discriminator, discriminator should give the 'fake data' low score
                // auto loss_d_f = torch::sum(-torch::log(torch::clamp_min(1.0 - probs_fake, 1e-6)));
                auto label_fake = torch::zeros_like(probs_fake);
                auto loss_d_f = F::binary_cross_entropy(probs_fake, label_fake);

                auto loss_G = Cfg::lambda_p * loss_p + Cfg::lambda_a * loss_g_a;
                auto loss_D = loss_d_r + loss_d_f;

                loss_G.backward();
                optim_G.step();

                optim_D.zero_grad();
                loss_D.backward();
                optim_D.step();

                // Log
                float loss_G_ = loss_G.detach().cpu().item().toFloat();
                float mask_loss_ = mask_loss.detach().cpu().item().toFloat();
                float depth_loss_ = depth_loss.detach().cpu().item().toFloat();
                float normal_loss_ = normal_loss.detach().cpu().item().toFloat();
                float loss_g_a_ = loss_g_a.detach().cpu().item().toFloat();
                float probs_g_ = probs_g.mean().detach().cpu().item().toFloat();
                float loss_D_ = loss_D.detach().cpu().item().toFloat();
                float probs_real_ = probs_real.mean().detach().cpu().item().toFloat();
                float probs_fake_ = probs_fake.mean().detach().cpu().item().toFloat();

                printf("[%2d/%2d][%3d] |G(%.4f) mask %.4f, depth %.4f, normal %.4f, adv %.4f, probs_g %.4f |D(%.4f) prob_real %.4f, prob_fake %.4f\n",
                       epoch, Cfg::epoch, step, loss_G_, mask_loss_, depth_loss_, normal_loss_, loss_g_a_, probs_g_, loss_D_, probs_real_, probs_fake_);
            }
            else
            {
                loss_p.backward();
                optim_G.step();

                // Log
                float loss_ = loss_p.detach().cpu().item().toFloat();
                float mask_loss_ = mask_loss.detach().cpu().item().toFloat();
                float depth_loss_ = depth_loss.detach().cpu().item().toFloat();
                float normal_loss_ = normal_loss.detach().cpu().item().toFloat();

                printf("[%2d/%2d][%3d] loss %.4f |mask %.4f |depth %.4f |normal %.4f\n",
                       epoch, Cfg::epoch, step, loss_, mask_loss_, depth_loss_, normal_loss_);
            }

            step += 1;
        }

        if (epoch % Cfg::val_epoch == 0)
        {
            printf("\n");
            torch::NoGradGuard no_grad;
            unet->eval();

            int iter = 0;
            for (auto &input : *val_loader)
            {

                std::vector<torch::Tensor> images;
                std::vector<torch::Tensor> gt_masks;
                std::vector<torch::Tensor> gt_depths;
                std::vector<torch::Tensor> gt_normals;
                std::vector<torch::Tensor> normal_masks;
                std::vector<ViewPred> gt_views;
                for (size_t i = 0; i < input.size(); i++)
                {
                    input[i].data.image = input[i].data.image.to(device);
                    input[i].target.gt_masks = input[i].target.gt_masks.to(device);
                    input[i].target.gt_depths = input[i].target.gt_depths.to(device);
                    input[i].target.gt_normals = input[i].target.gt_normals.to(device);
                    input[i].target.normal_masks = input[i].target.normal_masks.to(device);
                    input[i].target.targets = input[i].target.targets.to(device); // #views x 5 x H x W

                    images.push_back(input[i].data.image);
                    gt_masks.push_back(input[i].target.gt_masks); // #views x H x W
                    gt_depths.push_back(input[i].target.gt_depths);
                    gt_normals.push_back(input[i].target.gt_normals);     // (#views x 3) x H x W
                    normal_masks.push_back(input[i].target.normal_masks); // (#views x 3) x H x W
                    auto gt_view_vec = torch::split(input[i].target.targets, 1, 0);
                    gt_views.push_back(gt_view_vec);
                }
                //assert(gt_views.size() == Cfg::batch_size);

                auto img_batch = torch::stack(images); // N x 2 x H x W for input_views = "FS"
                auto gt_mask_batch = torch::stack(gt_masks);
                auto gt_depth_batch = torch::stack(gt_depths);
                auto gt_normal_batch = torch::stack(gt_normals);     // N x (#views x 3) x H x W
                auto normal_mask_batch = torch::stack(normal_masks); // N x (#views x 3) x H x W

                std::vector<torch::Tensor> view_preds;
                torch::Tensor pred_masks, pred_depths, pred_normals;
                std::tie(view_preds, pred_masks, pred_depths, pred_normals) = unet->forward(img_batch); // 2 x 5 x H x W

                pred_depths = torch::clamp(pred_depths, -1.0, 1.0); //clamp the predicted depths
                auto mask_loss = cal_mask_loss(gt_mask_batch, pred_masks);
                auto depth_loss = cal_depth_loss(gt_mask_batch, gt_depth_batch, pred_depths);
                auto normal_loss = cal_normal_loss(gt_depth_batch, normal_mask_batch, gt_normal_batch, pred_normals);
                auto loss_p = mask_loss + depth_loss + normal_loss;
                // Log
                float loss_ = loss_p.detach().cpu().item().toFloat();
                float mask_loss_ = mask_loss.detach().cpu().item().toFloat();
                float depth_loss_ = depth_loss.detach().cpu().item().toFloat();
                float normal_loss_ = normal_loss.detach().cpu().item().toFloat();

                printf("val [%2d/%2d][%3d] loss %.4f |mask %.4f |depth %.4f |normal %.4f\n",
                       epoch, Cfg::epoch, iter, loss_, mask_loss_, depth_loss_, normal_loss_);
                if (iter == 0)
                {
                    string save_dir = "output0916/epoch" + to_string(epoch);
                    if (!fs::exists(save_dir))
                    {
                        fs::create_directories(save_dir);
                    }

                    for (int view_id = 0; view_id < view_preds.size(); view_id++)
                    {
                        string mask_path = save_dir + "/" + to_string(view_id) + "_mask.png";
                        string depth_path = save_dir + "/" + to_string(view_id) + "_depth.png";
                        string normal_path = save_dir + "/" + to_string(view_id) + "_normal.png";
                        torch::Tensor pred = view_preds[view_id].detach().cpu();
                        std::vector<torch::Tensor> pred_vec = pred.split_with_sizes({1, 1, 3}, 1);
                        auto mask = pred_vec[0].squeeze(0);
                        auto depth = pred_vec[1].squeeze(0);
                        auto normal = pred_vec[2].squeeze(0);

                        cv::Mat mask_copy, depth_copy, normal_copy;
                        cv::Mat mask_img = ImgUtil::TensorToCvMat(mask);
                        mask_copy = mask_img.clone();
                        mask_img = ImgUtil::unnormalize_img(mask_img);
                        mask_img.convertTo(mask_img, CV_8UC1);

                        cv::Mat depth_img = ImgUtil::TensorToCvMat(depth);
                        depth_img = ImgUtil::unnormalize_img(depth_img, 65535.0);
                        depth_img.convertTo(depth_img, CV_16UC1);

                        cv::Mat normal_img = ImgUtil::TensorToCvMat(normal);
                        normal_img = ImgUtil::unnormalize_img(normal_img, 65535.0);
                        normal_img.convertTo(normal_img, CV_16UC3);

                        cv::imwrite(mask_path, mask_img);
                        cv::imwrite(depth_path, depth_img);
                        cv::imwrite(normal_path, normal_img);
                    }
                }
                iter++;
            }
            printf("\n\n");
        }

        if (epoch % 5 == 0)
        {
            char save_path[50];
            sprintf(save_path, "./output0916/unet%d.pt", epoch);
            torch::save(unet, save_path);
        }
    }

    torch::save(unet, "./output0916/unet_final.pt");
}

int main()
{
    train();
    return 0;
}