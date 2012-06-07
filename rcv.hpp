#ifndef RCV_HPP
#define RCV_HPP

#include <time.h>

namespace rcv
{
  //! Concatenate the left and the right images horizontally
  cv::Mat hcat(cv::Mat left, cv::Mat right, cv::Scalar fill = cv::Scalar(0));

  //! Concatenate the top and the bottom images vertically
  cv::Mat vcat(cv::Mat top, cv::Mat bottom, cv::Scalar fill = cv::Scalar(0));

  //! Plot the values as a simple line plot
  template<class T, class U>
  cv::Mat plot(T const * const values, size_t num_values, U max_value, cv::Size plot_size);

  //! A simple class for timing things.
  class Timer
  {
    public:
      //! Construct the timer - calls start() 
      Timer();

      //! Start timing
      void start();

      //! End timing, and elapsed time between start() and end() calls in seconds
      double end();

    private:
      timespec start_, end_; 
  };

  // ######################################################################
  cv::Mat hcat(cv::Mat left, cv::Mat right, cv::Scalar fill)
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
  cv::Mat vcat(cv::Mat top, cv::Mat bottom, cv::Scalar fill)
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
  Timer::Timer()
  {
    start();
  }

  // ######################################################################
  void Timer::start()
  {
    clock_gettime(CLOCK_REALTIME, &start_);
  }

  // ######################################################################
  double Timer::end()
  {
    clock_gettime(CLOCK_REALTIME, &end_);
    return (end_.tv_sec - start_.tv_sec) +
      (double)(end_.tv_nsec - start_.tv_nsec) / 1000000000.0;
  }

  // ######################################################################
  template<class T, class U>
  cv::Mat plot(T const * const values, size_t num_values, U max_value, cv::Size plot_size)
  {
    cv::Mat plot = cv::Mat::zeros(plot_size, CV_8UC1);

    int old_x;
    int old_y;
    for(size_t i=0; i<num_values; ++i)
    {
      int x = float(i)/float(num_values) * plot_size.width;
      int y = float(values[i]) / float(max_value) * plot_size.height;
      if(i == 0) { old_x = x; old_y = y; }

      cv::line(plot, cv::Point(old_x, plot_size.height - old_y), cv::Point(x, plot_size.height - y), cv::Scalar(255));
      old_x = x;
      old_y = y;
    }
    return plot;
  }

}
#endif // RCV_HPP
