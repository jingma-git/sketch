#pragma once
#include <torch/torch.h>
#include <string>

#include <boost/filesystem.hpp>
#include <opencv2/opencv.hpp>
namespace fs = boost::filesystem;

namespace net
{
    struct Window
    {
        int32_t y1{0};
        int32_t x1{0};
        int32_t y2{0};
        int32_t x2{0};
    };

    struct ImageMeta
    {
        int32_t image_id{0};
        int32_t image_width{0};
        int32_t image_height{0};
        Window window;
    };

    struct RawData
    {
        // record front-view image path if Cfg::input_views=="FS" or Cfg::input_views=="F"
        // record side-view image path if Cfg::input_views=="S"
        // record arbitrary-view image path if Cfg::input_views=="A"
        std::string img_path;

        // record side-view image path if Cfg::input_views=="FS"
        std::string img_path_side;

        //depth-normal paths for corresponding input image/images
        std::vector<std::string> dn_paths;
        char styleID;
    };

    struct Input
    {
        torch::Tensor image; // channels = #sketch_views
    };

    struct Target
    {
        torch::Tensor gt_masks; // Under 14 views
        torch::Tensor gt_depths;
        torch::Tensor gt_normals;
        torch::Tensor normal_masks;
        torch::Tensor targets; // 5xHxW for mask,depth,normal
    };

    using Sample = torch::data::Example<Input, Target>;

    class SketchDataset : public torch::data::Dataset<SketchDataset, Sample>
    {
    public:
        SketchDataset(std::string filename);

        Sample get(size_t index) override;
        torch::optional<size_t> size() const override;

    private:
        bool read_shape_list();
        void build_raw_data();
        bool parseSketch(const std::string imgName, cv::Mat &sketch);

        std::vector<std::string> shape_list;
        std::vector<RawData> raw_datas;

        std::string filename;
        fs::path data_dir;
        fs::path file_path;
    };
} // namespace net