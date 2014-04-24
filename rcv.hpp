#ifndef RCV_HPP
#define RCV_HPP

#include <time.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <type_traits>
#include <typeindex>
#include <stdexcept>

#include <iostream>

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

  //! Convert a CV_* type (from cv::Mat.type()) to a human readable string
  /*! Implementation ripped from http://stackoverflow.com/a/12336381/237092 */
  std::string type2string(int imgTypeInt)
  {
    int numImgTypes = 35; // 7 base types, with five channel options each (none or C1, ..., C4)

    int enum_ints[] = {
      CV_8U,  CV_8UC1,  CV_8UC2,  CV_8UC3,  CV_8UC4,
      CV_8S,  CV_8SC1,  CV_8SC2,  CV_8SC3,  CV_8SC4,
      CV_16U, CV_16UC1, CV_16UC2, CV_16UC3, CV_16UC4,
      CV_16S, CV_16SC1, CV_16SC2, CV_16SC3, CV_16SC4,
      CV_32S, CV_32SC1, CV_32SC2, CV_32SC3, CV_32SC4,
      CV_32F, CV_32FC1, CV_32FC2, CV_32FC3, CV_32FC4,
      CV_64F, CV_64FC1, CV_64FC2, CV_64FC3, CV_64FC4};

    static std::string const enum_strings[] = {
      "CV_8U",  "CV_8UC1",  "CV_8UC2",  "CV_8UC3",  "CV_8UC4",
      "CV_8S",  "CV_8SC1",  "CV_8SC2",  "CV_8SC3",  "CV_8SC4",
      "CV_16U", "CV_16UC1", "CV_16UC2", "CV_16UC3", "CV_16UC4",
      "CV_16S", "CV_16SC1", "CV_16SC2", "CV_16SC3", "CV_16SC4",
      "CV_32S", "CV_32SC1", "CV_32SC2", "CV_32SC3", "CV_32SC4",
      "CV_32F", "CV_32FC1", "CV_32FC2", "CV_32FC3", "CV_32FC4",
      "CV_64F", "CV_64FC1", "CV_64FC2", "CV_64FC3", "CV_64FC4"};

    for(int i=0; i<numImgTypes; i++)
    {
      if(imgTypeInt == enum_ints[i]) return enum_strings[i];
    }
    return "unknown image type";
  }

  //! Dispatch an image to a function which takes the underlying image data type as a template parameter
  /*! Example:
   * @code
   * template<class T>
   *   bool my_function(float param1, float param2, cv::Mat image, float param3)
   *   {
   *     assert(image.channels() == 1);
   *     return (image.at<T>(0,0) * param1 + param2 < param3);
   *   }
   *
   * cv::Mat my_unknown_matrix = getMatrixFromSomewhere();
   *
   * bool result = RCV_DISPATCH(my_unknown_matrix.type(), my_function,
   *   1.0, 2.0, my_unknown_matrix, 3.0);
   * @endcode
   * */
#define RCV_DISPATCH(type, function_name, ...)                                               \
  [&]() {                                                                                    \
    switch(CV_MAT_TYPE(type))                                                                \
    {                                                                                        \
      case CV_8U:  return function_name<uint8_t>(__VA_ARGS__);                               \
      case CV_8S:  return function_name<int8_t>(__VA_ARGS__);                                \
      case CV_16U: return function_name<uint16_t>(__VA_ARGS__);                              \
      case CV_16S: return function_name<int16_t>(__VA_ARGS__);                               \
      case CV_32S: return function_name<int32_t>(__VA_ARGS__);                               \
      case CV_32F: return function_name<float>(__VA_ARGS__);                                 \
      case CV_64F: return function_name<double>(__VA_ARGS__);                                \
      default: throw std::runtime_error("Unsupported data type: " + rcv::type2string(type)); \
    };                                                                                       \
  }()
  
  //! Dispatch an image to a function which takes the underlying image data type as a template parameter and has no return type
  /*! \see RCV_DISPATCH for an example */
#define RCV_DISPATCH_NO_RETURN(type, function_name, ...)                                     \
  [&]() {                                                                                    \
    switch(CV_MAT_TYPE(type))                                                                \
    {                                                                                        \
      case CV_8U:  function_name<uint8_t>(__VA_ARGS__);   break;                             \
      case CV_8S:  function_name<int8_t>(__VA_ARGS__);    break;                             \
      case CV_16U: function_name<uint16_t>(__VA_ARGS__);  break;                             \
      case CV_16S: function_name<int16_t>(__VA_ARGS__);   break;                             \
      case CV_32S: function_name<int32_t>(__VA_ARGS__);   break;                             \
      case CV_32F: function_name<float>(__VA_ARGS__);     break;                             \
      case CV_64F: function_name<double>(__VA_ARGS__);    break;                             \
      default: throw std::runtime_error("Unsupported data type: " + rcv::type2string(type)); \
    };                                                                                       \
  }()

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
  //! Concatenate a set of images horizontally
  cv::Mat hcat(std::vector<cv::Mat> const & images, cv::Scalar fill = cv::Scalar(0))
  {
    if(images.empty()) return cv::Mat();
    if(images.size() == 1) return images[0];

    assert(std::all_of(images.begin(), images.end(),
          [images](cv::Mat const & image) { return image.type() == images[0].type(); }));

    int const rows = std::max_element(images.begin(), images.end(), [](cv::Mat a, cv::Mat b) { return a.rows < b.rows; })->rows;
    int cols = std::accumulate(images.begin(), images.end(), 0, [](int n, cv::Mat m) { return n+m.cols; });

    cv::Mat ret(rows, cols, images[0].type(), fill);

    int c = 0;
    for(cv::Mat const & image : images)
    {
      cv::Mat roi = ret(cv::Rect(c, 0, image.cols, image.rows));
      image.copyTo(roi);
      c += image.cols;
    }

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
  //! Concatenate a set of images vertically
  cv::Mat vcat(std::vector<cv::Mat> const & images, cv::Scalar fill = cv::Scalar(0))
  {
    if(images.empty()) return cv::Mat();
    if(images.size() == 1) return images[0];

    assert(std::all_of(images.begin(), images.end(),
          [images](cv::Mat const & image) { return image.type() == images[0].type(); }));

    int rows = std::accumulate(images.begin(), images.end(), 0, [](int n, cv::Mat m) { return n+m.rows; });
    int const cols = std::max_element(images.begin(), images.end(), [](cv::Mat a, cv::Mat b) { return a.cols < b.cols; })->cols;

    cv::Mat ret(rows, cols, images[0].type(), fill);

    int r = 0;
    for(cv::Mat const & image : images)
    {
      cv::Mat roi = ret(cv::Rect(0, r, image.cols, image.rows));
      image.copyTo(roi);
      r += image.rows;
    }


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

  // ######################################################################
  //! Colorize the input using Dave Green's 'cubehelix' algorithm
  /*! This implementation based on Jim Davenport's python implementation found here: https://github.com/jradavenport/cubehelix/ */
  class cubehelix
  {
    public:

      struct create
      {
        create() :
          nlev_(256), start_(0.5), rot_(-1.5), gamma_(1.0), hue_(1.2), reverse_(false) 
        { }

        operator cubehelix()
        {
          return cubehelix(nlev_, start_, rot_, gamma_, hue_, reverse_);
        }

        create & nlev(size_t val)  { nlev_    = val; return *this; }
        create & start(float val)  { start_   = val; return *this; }
        create & rot(float val)    { rot_     = val; return *this; }
        create & gamma(float val)  { gamma_   = val; return *this; }
        create & hue(float val)    { hue_     = val; return *this; }
        create & reverse()         { reverse_ = !reverse_; return *this; }

        size_t nlev_;
        float start_, rot_, gamma_, hue_;
        bool reverse_;
      };


      //! Construct a cubehelix object and initialize it's mapping tables
      /*! 
        @param nlev The number of color levels in the color map.

        @param start The starting position in the color space. 0=blue, 1=red, 2=green. Defaults to 0.5.

        @param rot The number of rotations through the rainbow. Can be positive 
        or negative, indicating direction of rainbow. Negative values
        correspond to Blue->Red direction.

        @param gamma The gamma correction for intensity.

        @param hue The hue intensity factor.

        @param reverse Set to True to reverse the color map. Will go from black to
        white. Good for density plots where shade~density.
        */
      cubehelix(size_t nlev=256, float start=0.5, float rot=-1.5, float gamma=1.0, float hue=1.2, bool reverse=false) :
        nlev(nlev)
    {
      // Set up the parameters
      std::vector<double> fract(nlev);
      std::iota(fract.begin(), fract.end(), 0);
      for(auto & f : fract) f /= (nlev-1.0);

      std::vector<double> angle(nlev);
      std::transform(fract.begin(), fract.end(), angle.begin(),
          [start, rot](double f) { return 2*M_PI * (start/3.0 + 1.0 + rot*f); });

      for(auto & f : fract) f = std::pow(f, gamma);

      std::vector<double> amp(nlev);
      std::transform(fract.begin(), fract.end(), amp.begin(),
          [hue](double f) { return hue * f * (1.0-f)/2.0; });

      // compute the RGB vectors according to main equations
      red_.resize(nlev);
      grn_.resize(nlev);
      blu_.resize(nlev);
      for(size_t i=0; i<nlev; ++i)
      {
        double const s = std::sin(angle[i]);
        double const c = std::cos(angle[i]);
        double const f = fract[i];
        double const a = amp[i];
        red_[i] = std::max(0.0, std::min( 255.0, std::round((f+a*(-0.14861*c + 1.78277*s))*255.0)));
        grn_[i] = std::max(0.0, std::min( 255.0, std::round((f+a*(-0.29227*c - 0.90649*s))*255.0)));
        blu_[i] = std::max(0.0, std::min( 255.0, std::round((f+a*(1.97294*c))*255.0)));
      }


      if(reverse)
      {
        std::reverse(red_.begin(), red_.end());
        std::reverse(grn_.begin(), grn_.end());
        std::reverse(blu_.begin(), blu_.end());
      }
    }

      //! Map the values of an input image to cubehelix colors
      cv::Mat operator()(cv::Mat const & input)
      {
        if(input.channels() != 1)
          throw std::runtime_error(
              "Too many channels (" + std::to_string(input.channels()) + ") in input image");

        return RCV_DISPATCH(input.type(), process, input);

        //switch(CV_MAT_TYPE(input.type()))
        //{
        //  case CV_8U:  return process<uint8_t>(input);
        //  case CV_8S:  return process<int8_t>(input);
        //  case CV_16U: return process<uint16_t>(input);
        //  case CV_16S: return process<int16_t>(input);
        //  case CV_32S: return process<int32_t>(input);
        //  case CV_32F: return process<float>(input);
        //  case CV_64F: return process<double>(input);
        //  default: throw std::runtime_error("Unsupported data type: " +
        //               std::to_string(CV_MAT_TYPE(input.type())));
        //};
      }

   private:
      template<class T>
      cv::Mat process(cv::Mat const & input)
      {
        double minv, maxv;
        cv::minMaxLoc(input, &minv, &maxv);
        std::cout << "minmax: " << minv << ", " << maxv << std::endl;

        if(minv == maxv) return cv::Mat::zeros(input.rows, input.cols, CV_8UC3);
        cv::Mat ret(input.size(), CV_8UC3);

        std::transform(input.begin<T>(), input.end<T>(), ret.begin<cv::Vec3b>(),
            [this, minv, maxv](T const v) 
            { 
              size_t const idx = (v - minv) / (maxv - minv) * (nlev - 1);
              return cv::Vec3b(red_[idx], blu_[idx], grn_[idx]);
            });

        return ret;
      }

      size_t const nlev;
      std::vector<uint8_t> red_, blu_, grn_;
  };

}
#endif // RCV_HPP

