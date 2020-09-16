#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <CvUtil.h>

#include "SketchDataset.h"
#include "Config.h"
#include "ImgUtil.h"

using namespace net;
using namespace std;

SketchDataset::SketchDataset(std::string filename) : filename(filename)
{
    data_dir = fs::path(Cfg::data_dir);
    file_path = data_dir / filename;
    read_shape_list();

    if (Cfg::style_ids == "c")
    {
        Cfg::in_channels = Cfg::input_views.size() * 3;
    }

    build_raw_data();
}

bool SketchDataset::read_shape_list()
{
    if (!fs::exists(file_path))
    {
        cerr << __FILE__ << " " << __LINE__ << ":" << file_path << " does not exists" << endl;
        return false;
    }
    ifstream file(file_path.c_str());
    string line;
    while (getline(file, line))
    {
        shape_list.push_back(line);
    }

    return true;
}

void SketchDataset::build_raw_data()
{
    for (size_t i = 0; i < shape_list.size(); i++)
    {
        string &shape_name = shape_list[i];
        for (size_t j = 0; j < Cfg::style_ids.size(); j++)
        {
            RawData raw_data;
            char styleID = Cfg::style_ids[j];
            raw_data.styleID = styleID;

            if (Cfg::input_views == "FS")
            {
                char img_name_front[50], img_name_side[50];
                sprintf(img_name_front, "sketch-F-%c.png", styleID);
                sprintf(img_name_side, "sketch-S-%c.png", styleID);
                raw_data.img_path = (data_dir / "sketch" / shape_name / img_name_front).c_str();
                raw_data.img_path_side = (data_dir / "sketch" / shape_name / img_name_side).c_str();

                string target_name_front = (data_dir / "dnfs" / shape_name / "dnfs-256-0.png").c_str();
                string target_name_side = (data_dir / "dnfs" / shape_name / "dnfs-256-1.png").c_str();
                raw_data.dn_paths.push_back(target_name_front);
                raw_data.dn_paths.push_back(target_name_side);
            }
            else if (Cfg::input_views == "F")
            {
                char img_name[50];
                sprintf(img_name, "sketch-F-%c.png", styleID);
                raw_data.img_path = (data_dir / "sketch" / shape_name / img_name).c_str();

                string target_name_front = (data_dir / "dnfs" / shape_name / "dnfs-256-0.png").c_str();
                raw_data.dn_paths.push_back(target_name_front);
            }
            else if (Cfg::input_views == "S")
            {
                char img_name[50];
                sprintf(img_name, "sketch-S-%c.png", styleID);
                raw_data.img_path = (data_dir / "sketch" / shape_name / img_name).c_str();

                string target_name_side = (data_dir / "dnfs" / shape_name / "dnfs-256-1.png").c_str();
                raw_data.dn_paths.push_back(target_name_side);
            }
            else
            {
                // ToDO: read all 33 views:
                // 3*7(for front, right side, left side, each have 7 image corresponding to -30-30 [step is 10])
                // 12 views around icohedron for 3D point cloud prediction
                // so totally there are 33 views
            }

            for (int j = 0; j < Cfg::num_target_views; j++)
            {
                char target_name[50];
                sprintf(target_name, "dn-256-%d.png", j);
                string dn_path = (data_dir / "dn" / shape_name / target_name).c_str();
                raw_data.dn_paths.push_back(dn_path);
            }

            raw_datas.push_back(raw_data);
        }
    }
}

Sample SketchDataset::get(size_t index)
{
    const RawData &raw_data = raw_datas[index];
    Sample result;

    cv::Mat img, img_side;
    if (Cfg::input_views == "FS")
    {
        if (!parseSketch(raw_data.img_path, img))
        {
            cerr << __FILE__ << " " << __LINE__ << ": " << raw_data.img_path << " doesn't exist" << endl;
            exit(-1);
        }

        if (!parseSketch(raw_data.img_path_side, img_side))
        {
            cerr << __FILE__ << " " << __LINE__ << ": " << raw_data.img_path_side << " doesn't exist" << endl;
            exit(-1);
        }

        img = ImgUtil::normalize_img(img);
        img_side = ImgUtil::normalize_img(img_side);
        auto img_tensor = ImgUtil::CvImageToTensor(img);
        auto img_side_tensor = ImgUtil::CvImageToTensor(img_side);
        result.data.image = torch::stack({img_tensor, img_side_tensor});
    }
    else
    {
        if (!parseSketch(raw_data.img_path, img))
        {
            cerr << __FILE__ << " " << __LINE__ << ": " << raw_data.img_path << " doesn't exist" << endl;
            exit(-1);
        }
        img = ImgUtil::normalize_img(img);
        auto img_tensor = ImgUtil::CvImageToTensor(img);
        result.data.image = torch::stack({img_tensor});
    }

    std::vector<torch::Tensor> mask_tensors;
    std::vector<torch::Tensor> normal_tensors;
    std::vector<torch::Tensor> depth_tensors;
    std::vector<torch::Tensor> normal_masks;
    std::vector<torch::Tensor> target_vec;
    for (size_t i = 0; i < raw_data.dn_paths.size(); i++)
    {
        cv::Mat dn_img = cv::imread(raw_data.dn_paths[i], cv::IMREAD_UNCHANGED);
        dn_img = ImgUtil::normalize_img(dn_img);

        cv::Mat mask = ImgUtil::extract_boolean_mask(dn_img);
        auto mask_tensor = ImgUtil::CvImageToTensor(mask);
        mask_tensors.push_back(mask_tensor);                                           // H x W
        normal_masks.push_back(torch::stack({mask_tensor, mask_tensor, mask_tensor})); // 3 x H x W

        cv::Mat normal, depth;
        ImgUtil::extract_depth_normal(dn_img, normal, depth);
        auto depth_tensor = ImgUtil::CvImageToTensor(depth);
        auto normal_tensor = ImgUtil::CvImageToTensor(normal); //  3 x H x W
        depth_tensors.push_back(depth_tensor);
        normal_tensors.push_back(normal_tensor);

        std::vector<torch::Tensor> tmp_target({mask_tensor.unsqueeze(0), depth_tensor.unsqueeze(0), normal_tensor});
        auto target = torch::cat(tmp_target, 0);
        target_vec.push_back(target);
    }
    result.target.gt_masks = torch::stack(mask_tensors);
    result.target.gt_depths = torch::stack(depth_tensors);    // #views x H x W
    result.target.gt_normals = torch::cat(normal_tensors, 0); // (#views x 3) x H x W
    result.target.normal_masks = torch::cat(normal_masks, 0);
    result.target.targets = torch::stack(target_vec);

    // cout << raw_data.img_path << ":" << CvUtil::type2str(img.type()) << "->" << result.data.image.sizes() << "-" << result.data.image.type() << endl;
    // cout << "mask: " << result.target.gt_masks.sizes() << endl;
    // cout << "depth: " << result.target.gt_depths.sizes() << endl;
    // cout << "normal: " << result.target.gt_normals.sizes() << endl;
    // cout << "target: " << result.target.targets.sizes() << endl;

    return result;
}

torch::optional<size_t> SketchDataset::size() const
{
    return raw_datas.size();
}

bool SketchDataset::parseSketch(const std::string imgName, cv::Mat &sketch)
{
    if (!fs::exists(imgName))
        return false;

    sketch = cv::imread(imgName, cv::IMREAD_UNCHANGED);

    if (sketch.channels() == 3)
    {
        cv::cvtColor(sketch, sketch, cv::COLOR_BGR2GRAY);
    }
    return true;
}