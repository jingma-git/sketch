#include "Config.h"
using namespace net;
using namespace std;

// Idea: can I give the network an contour in arbitray view, and predict contour in other views
// Input: image in arbitrary view, Output: (mask, depth, normal in 14 views) or (contour in 14 views and input's views' orientation)

// Class: 1. Character with legs and arms
//        2. Side-view animal
//        3. One-stroke Sketch: Birds, Fish // the easiest one, do not consider now
//        4. Character with clothes
string Cfg::data_dir = "./data/Character";

string Cfg::input_views = "FS"; // Front F, Side S, Front and Side FS, Arbitrary view A (choose from 0-11, maybe..., For EasyToy Dataset)
string Cfg::style_ids = "0123"; // digit 0-9 is reserved for sketch image, 'c' is reserved for colorful image
int Cfg::num_target_views = 12;
int Cfg::num_views = Cfg::input_views.size() + Cfg::num_target_views;
int Cfg::out_channels = 5; // channel 0: mask, channel 1 depth, channel 2-4 normal
int Cfg::in_channels = input_views.size();
bool Cfg::is_scale_depth_loss = true;

ConvType Cfg::conv_type = ConvType::DE_CONV;
std::string Cfg::activation_fn = "leaky_relu"; //relu,leaky_relu,tanh

// Train
int Cfg::epoch = 100;
int Cfg::val_epoch = 1;
float Cfg::lr = 1e-4;
float Cfg::weight_decay = 1e-5;
int Cfg::batch_size = 2;
bool Cfg::with_adversarial = false;
bool Cfg::with_normal = true;
float Cfg::lambda_p = 1.0;  // weight for Generator's prediction loss
float Cfg::lambda_a = 0.01; // weight for Generator's adversarial loss
