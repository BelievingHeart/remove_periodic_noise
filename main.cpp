#include <fmt/format.h>
#include <opencv2/opencv.hpp>
#include <tuple>

namespace db {
void imshow(const cv::String &winName_, cv::Mat img_) {
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
static cv::Mat remove_periodic_noise(cv::Mat &input_image, const cv::Point2i noise_spike);

int main() {
  cv::Mat input_image = cv::imread(
      "/home/afterburner/Pictures/periodic_noise.jpg", cv::IMREAD_GRAYSCALE);
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
  db::imshow(winName, magnitude);
  cv::waitKey(0);

  cv::Mat recovered_image = remove_periodic_noise(complex, noise_spike);
  db::imshow("recovered image", recovered_image);
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
  magnitude.at<float>(0) = 0;
  return {magnitude, complex};
}

cv::Mat remove_periodic_noise(cv::Mat &input_image, const cv::Point2i noise_spike) {
  cv::Point2i c2, c3, c4;
  c2.x = noise_spike.x;
  c2.y = input_image.rows - noise_spike.y;
  c3.x = input_image.cols - noise_spike.x;
  c3.y = noise_spike.y;
  c4.x = input_image.cols - noise_spike.x;
  c4.y = input_image.rows - noise_spike.y;
  cv::Mat anti_spike = cv::Mat::ones(input_image.size(), CV_32FC1);
  for (const auto &pt : {noise_spike, c2, c3, c4}) {
    cv::circle(anti_spike, pt, 10, {0}, -1);
  }
  std::vector<cv::Mat> channles{
      std::move(anti_spike),
      cv::Mat::zeros(input_image.size(), CV_32FC1)};
  cv::Mat anti_spike_composite;
  cv::merge(channles, anti_spike_composite);
  cv::mulSpectrums(input_image, anti_spike_composite, input_image, cv::DFT_ROWS);
  cv::idft(input_image, input_image);
  cv::split(input_image, channles);
  return channles[0];
}
