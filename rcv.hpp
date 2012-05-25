#ifndef RCV_HPP
#define RCV_HPP

namespace rcv
{
  //! Concatenate the left and the right images horizontally
  cv::Mat hcat(cv::Mat left, cv::Mat right, cv::Scalar fill = cv::Scalar(0));

  //! Concatenate the top and the bottom images vertically
  cv::Mat vcat(cv::Mat top, cv::Mat bottom, cv::Scalar fill = cv::Scalar(0));


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
}
#endif // RCV_HPP
