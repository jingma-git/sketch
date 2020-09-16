#pragma once
#include <torch/torch.h>

namespace net
{
    // calculate mask loss
    torch::Tensor cal_mask_loss(torch::Tensor gt_mask, torch::Tensor pred_mask);
    torch::Tensor cal_depth_loss(torch::Tensor gt_mask, torch::Tensor gt_depth, torch::Tensor pred_depth);
    torch::Tensor cal_normal_loss(torch::Tensor gt_mask, torch::Tensor normal_mask, torch::Tensor gt_normal, torch::Tensor pred_normal);
} // namespace net