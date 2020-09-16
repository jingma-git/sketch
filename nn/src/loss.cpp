#include "loss.h"
#include "Config.h"

namespace F = torch::nn::functional;
namespace net
{
    torch::Tensor cal_mask_loss(torch::Tensor gt_mask, torch::Tensor pred_mask)
    {
        if (Cfg::activation_fn == "tanh")
        {
            pred_mask = pred_mask * 0.5 + 0.5; //map [-1, 1] to [0, 1]
        }
        auto loss = F::binary_cross_entropy_with_logits(pred_mask, gt_mask);
        if (Cfg::is_scale_depth_loss)
        {
            loss = loss * gt_mask.size(0) * gt_mask.size(1);
        }
        return loss;
    }
    torch::Tensor cal_depth_loss(torch::Tensor gt_mask, torch::Tensor gt_depth, torch::Tensor pred_depth)
    {
        auto binary_nums = torch::sum(gt_mask);

        auto loss = F::l1_loss(pred_depth * gt_mask, gt_depth * gt_mask, F::L1LossFuncOptions().reduction(torch::kSum)) / binary_nums;
        if (Cfg::is_scale_depth_loss)
        {
            loss = loss * gt_depth.size(0) * gt_depth.size(1);
        }
        return loss;
    }

    torch::Tensor cal_normal_loss(torch::Tensor gt_mask, torch::Tensor normal_mask, torch::Tensor gt_normal, torch::Tensor pred_normal)
    {
        // with unit length 1-n_1*n_2 = 0.5*||n_1-n_2||^2

        auto binary_nums = torch::sum(gt_mask);                                                    // N x (#views) x H x W
        auto loss = 0.5 * torch::abs((gt_normal - pred_normal) * normal_mask).sum() / binary_nums; // Note: Could change to square

        if (Cfg::is_scale_depth_loss)
        {
            loss = loss * gt_normal.size(0) * gt_normal.size(1);
        }
        return loss;
    }
} // namespace net
