#ifndef RCV_HPP
#define RCV_HPP

#include <time.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <type_traits>
#include <typeindex>

namespace rcv
{
  //! Get the CV_ flag for a given numeric type
  /*! E.g. type2cv<int16_t>::value will resolve to CV_16S */
  template<class T> struct type2cv {};
  template<> struct type2cv<uint8_t>  { static const int value = CV_8U;  };
  template<> struct type2cv<int8_t>   { static const int value = CV_8S;  };
  template<> struct type2cv<uint16_t> { static const int value = CV_16U; };
  template<> struct type2cv<int16_t>  { static const int value = CV_16S; };
  template<> struct type2cv<int32_t>  { static const int value = CV_32S; };
  template<> struct type2cv<float>    { static const int value = CV_32F; };
  template<> struct type2cv<double>   { static const int value = CV_64F; };

  // ######################################################################
  //! Concatenate the left and the right images horizontally
  cv::Mat hcat(cv::Mat left, cv::Mat right, cv::Scalar fill = cv::Scalar(0))
  {
    assert(left.type() == right.type());

    int const rows = std::max(left.rows, right.rows);
    int const cols = left.cols + right.cols;
    cv::Mat ret(rows, cols, left.type(), fill); 

    cv::Mat left_ret_roi = ret(cv::Rect(0, 0, left.cols, left.rows));
    left.copyTo(left_ret_roi);

    cv::Mat right_ret_roi = ret(cv::Rect(left.cols, 0, right.cols, right.rows));
    right.copyTo(right_ret_roi);

    return ret;
  }

  // ######################################################################
  //! Concatenate the top and the bottom images vertically
  cv::Mat vcat(cv::Mat top, cv::Mat bottom, cv::Scalar fill = cv::Scalar(0))
  {
    assert(top.type() == bottom.type());

    int const rows = top.rows + bottom.rows;
    int const cols = std::max(top.cols, bottom.cols);
    cv::Mat ret(rows, cols, top.type(), fill); 

    cv::Mat top_ret_roi = ret(cv::Rect(0, 0, top.cols, top.rows));
    top.copyTo(top_ret_roi);

    cv::Mat bottom_ret_roi = ret(cv::Rect(0, top.rows, bottom.cols, bottom.rows));
    bottom.copyTo(bottom_ret_roi);

    return ret;
  }

  // ######################################################################
  class AutoScaleIndicator { };
  AutoScaleIndicator autoscale;

  template<class Iterator, class MaxScaleValue> 
    auto get_max_value(Iterator const begin, Iterator const end, MaxScaleValue max_value __attribute__ ((unused))) -> 
    typename std::enable_if<std::is_same<MaxScaleValue, AutoScaleIndicator>::value, typename std::remove_reference<decltype(*begin)>::type>::type
    { return *std::max_element(begin, end); }

  template<class Iterator, class MaxScaleValue> 
    auto get_max_value(Iterator const begin  __attribute__ ((unused)), Iterator const end  __attribute__ ((unused)), MaxScaleValue max_value) -> 
    typename std::enable_if<!std::is_same<MaxScaleValue, AutoScaleIndicator>::value, typename std::remove_reference<decltype(*begin)>::type>::type
    { return max_value; }

  // ######################################################################
  //! Plot the values as a simple line plot
  template<class Iterator, class MinScaleValue=int, class MaxScaleValue=AutoScaleIndicator>
  cv::Mat plot(Iterator const begin, Iterator const end, cv::Size plot_size,
      MinScaleValue min_value = 0, MaxScaleValue max_value = autoscale,
      cv::Scalar line_color=cv::Scalar(255), int line_width=1)
  {
    typedef typename std::remove_reference<decltype(*begin)>::type T;

    cv::Mat plot = cv::Mat::zeros(plot_size, CV_8UC3);

    T min_value_ = min_value;
    T max_value_ = get_max_value(begin, end, max_value);

    int old_x = 0;
    int old_y = 0;
    size_t const num_values = std::distance(begin, end);
    Iterator it = begin;
    for(size_t i=0; i<num_values; ++i, ++it)
    {
      int x = float(i)/float(num_values) * plot_size.width;
      int y = (float(*it - min_value_) / float(max_value_ - min_value_)) * plot_size.height;
      y = std::max(0, std::min(plot_size.height-1, y));

      cv::line(plot, cv::Point(old_x, plot_size.height - old_y - 1), cv::Point(x, plot_size.height - y - 1), line_color, line_width);
      old_x = x;
      old_y = y;
    }
    return plot; 
  }

}
#endif // RCV_HPP

