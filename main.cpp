#include <fmt/core.h>                    // for print
#include <opencv2/core/hal/interface.h>  // for CV_8U, CV_32FC1
#include <opencv2/core.hpp>              // for normalize, merge, split, dft
#include <opencv2/highgui.hpp>           // for imshow, waitKey, namedWindow
#include <opencv2/imgproc.hpp>           // for circle

namespace db {
void imshow(const cv::String &winName_, const cv::Mat& img_) {
#ifdef DEBUG
  if (img_.depth() == CV_8U) {
    cv::imshow(winName_, img_);
  } else {
    double min_ = 0, max_ = 0;
    cv::Point2i min_pt, max_pt;
    cv::Mat temp;
    cv::minMaxLoc(img_, &min_, &max_, &min_pt, &max_pt, cv::noArray());
    cv::normalize(img_, temp, 0, 255, cv::NORM_MINMAX, CV_8U, {});

    cv::imshow(winName_, temp);
    if (min_ < 0 || max_ > 1) {
      fmt::print("[DEBUG] {} is not of any showing formats, a makeshift image "
                 "is create.\nOriginal states:\n",
                 winName_);
      fmt::print("minVal: {} at point ({},{})\n", min_, min_pt.x, min_pt.y);
      fmt::print("maxVal: {} at point ({},{})\n\n", max_, max_pt.x, max_pt.y);
    }
  }
#endif // DEBUG
}
}; // namespace db



static std::tuple<cv::Mat, cv::Mat> calcPSD(const cv::Mat &input_image);
static cv::Mat remove_periodic_noise(cv::Mat &&complex_, const cv::Point2i &noise_spike);

int main(const int argc, const char* argv[]) {
  if (argc < 2) {
    fmt::print("Required a path to an image");
    return 0;
  }
  cv::Mat input_image = cv::imread(
      argv[1], cv::IMREAD_GRAYSCALE);
  if (input_image.empty()) {
    fmt::print("Error reading image: {}", argv[1]);
    return 1;
  }
  cv::imshow("input_image", input_image);
  cv::Rect roi(0, 0, input_image.cols & -2, input_image.rows & -2);
  input_image = input_image(roi);

  auto [magnitude, complex] = calcPSD(input_image);
  const char *winName = "magnitude";
  cv::Point2i noise_spike{};
  cv::namedWindow(winName);
  cv::setMouseCallback(winName,
          [](int event, int x, int y, int flags, void *pt) {
    if(event == cv::EVENT_LBUTTONDOWN){
      auto spike_ptr = static_cast<cv::Point2i *>(pt);
      spike_ptr->x = x;
      spike_ptr->y = y;
      fmt::print("Selected frequency spike at ({},{}), now press any key to continue.\n", x, y);
    }
    },
          &noise_spike);
  cv::normalize(magnitude, magnitude, 0, 255, cv::NORM_MINMAX, CV_8U, {});
  cv::imshow("magnitude", magnitude);
  cv::waitKey(0);

  cv::Mat recovered_image = remove_periodic_noise(std::move(complex), noise_spike);
  cv::normalize(recovered_image, recovered_image, 0, 255, cv::NORM_MINMAX, CV_8U, {});
  cv::imshow("recovered image", recovered_image);
  cv::waitKey(0);
  return 0;
}

std::tuple<cv::Mat, cv::Mat> calcPSD(const cv::Mat &input_image) {
  std::vector<cv::Mat> channels{cv::Mat_<float>(input_image),
                                cv::Mat::zeros(input_image.size(), CV_32FC1)};
  cv::Mat composite;
  cv::merge(channels, composite);
  cv::Mat complex;
  cv::dft(composite, complex, cv::DFT_COMPLEX_OUTPUT);
  cv::split(complex, channels);
  cv::Mat magnitude;
  cv::magnitude(channels[0], channels[1], magnitude);
  cv::pow(magnitude, 2, magnitude);
  magnitude.at<float>(0) = 0; //NOTE: very important, value at (0,0) is the main frequency of the whole image. For visualization reason, it must set to zero, otherwise it will dwarf all the noise spikes.
  return {magnitude, complex};
}

cv::Mat remove_periodic_noise(cv::Mat &&complex_, const cv::Point2i &noise_spike) {
  cv::Point2i c2, c3, c4;
  c2.x = noise_spike.x;
  c2.y = complex_.rows - noise_spike.y;
  c3.x = complex_.cols - noise_spike.x;
  c3.y = noise_spike.y;
  c4.x = complex_.cols - noise_spike.x;
  c4.y = complex_.rows - noise_spike.y;
  cv::Mat anti_spike = cv::Mat::ones(complex_.size(), CV_32FC1);
  for (const auto &pt : {noise_spike, c2, c3, c4}) {
    cv::circle(anti_spike, pt, 10, {0}, -1);
  }
  std::vector<cv::Mat> channels{
      std::move(anti_spike),
      cv::Mat::zeros(complex_.size(), CV_32FC1)};
  cv::Mat anti_spike_composite;
  cv::merge(channels, anti_spike_composite);
  cv::mulSpectrums(complex_, anti_spike_composite, complex_, cv::DFT_ROWS);
  cv::idft(complex_, complex_);
  cv::split(complex_, channels);
  return channels[0];
}
