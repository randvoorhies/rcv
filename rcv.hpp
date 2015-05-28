#ifndef RCV_HPP
#define RCV_HPP

#include <time.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <type_traits>
#include <typeindex>
#include <stdexcept>
#include <iomanip>
#include <sstream>

#include <iostream>
#include <numeric>

namespace rcv
{

  /*! \defgroup TypeUtils Type Utilities 
      Various utilities to help out with the inherent type unsafety of OpenCV
      @{ */
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
  /*! For example:
   * @code
   * cv::Mat mat = poorly_documented_function();
   * std::cout << "My matrix is of type: " << rcv::type2string(mat.type()) << std::endl;
   * @endcode
   * \note Implementation ripped from http://stackoverflow.com/a/12336381/237092 */
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

  //! Get the standard "white" value for a given OpenCV data type
  /*! For example, CV_8U images typically use 255 as their white value, while CV32F images typically use 1.0
   * @code
   * double white = rcv::white_value(myMat.type());
   * @endcode
   * */
  double constexpr white_value(int mat_type)
  {
    return CV_MAT_TYPE(mat_type) == CV_8U ? 255.0 :
      (
       CV_MAT_TYPE(mat_type) == CV_16U ? 65536.0 : 
       (
        1.0
       )
      );
  }

  //! Convert a cv::Mat, and scale it to the typical range for the destination type
  /*! \sa white_value */
  cv::Mat convert_and_scale(cv::Mat const & image, int const rtype)
  {
    cv::Mat result;
    image.convertTo(result, rtype, white_value(rtype) / white_value(image.type()));
    return result;
  }

  //! Dispatch an image to a function which takes the underlying image data type as a template parameter
  /*! This is helpful e.g. when you need to access the pixel values of an
   * image, but you don't know that image's type. Using the .at() method
   * requires a template parameter, so RCV_DISPATCH can be used to call the
   * proper templated method.
   
   * Example:
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
   *
   * \todo Implement this without macros
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

  /*! @} */

  /*! \defgroup ImageConcatentation Image Concatenation
      Paste two or more images together, generally for display purposes

      \warning These methods are totally unoptimized. A real implementation
      would use lazy evaluation to make one copy of the pixels at the last
      moment. This is not implemented yet, so do not use in performance
      sensitive code.

      @{
  */

  // ######################################################################
  //! Concatenate the left and the right images horizontally
  /*! Use this version to concatenate two images
   *
   * @param left The left image
   * @param right The right image
   * @param fill A color to fill in the background of an image if it is smaller than the other
   */
  cv::Mat hcat(cv::Mat left, cv::Mat right, cv::Scalar fill = cv::Scalar(0))
  {
    if(left.type() != right.type())
      throw std::runtime_error("In rcv::hcat: mismatched types between "
          "left (" + rcv::type2string(left.type()) + ") "
          "and right (" + rcv::type2string(right.type()) + ")");

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
  /*! Use this version to concatenate more than two images, e.g.
   *
   * Example:
   * @code
   * cv::Mat display = rcv::hcat({first_image, second_image, third_image, fourth_image});
   * @endcode
   *
   * @param images A list of images to be concatenated in left-to-right order
   * @param fill A color to fill in the background of an image if it is smaller than the other
   */
  cv::Mat hcat(std::vector<cv::Mat> const & images, cv::Scalar fill = cv::Scalar(0))
  {
    if(images.empty()) return cv::Mat();
    if(images.size() == 1) return images[0];

    for(size_t i=1; i<images.size(); ++i)
      if(images[i].type() != images[0].type())
        throw std::runtime_error("In rcv::hcat: mismatched types between "
            "images[0] (" + rcv::type2string(images[0].type()) + ") "
            "and images[" + std::to_string(i) + "] (" + rcv::type2string(images[i].type()) + ")");

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
  /*! Use this version to concatenate two images
   *
   * @param top The top image
   * @param bottom The bottom image
   * @param fill A color to fill in the background of an image if it is smaller than the other
   */
  cv::Mat vcat(cv::Mat top, cv::Mat bottom, cv::Scalar fill = cv::Scalar(0))
  {
    if(top.type() != bottom.type())
      throw std::runtime_error("In rcv::vcat: mismatched types between "
          "top (" + rcv::type2string(top.type()) + ") "
          "and bottom (" + rcv::type2string(bottom.type()) + ")");

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
  /*! Use this version to concatenate more than two images, e.g.
   *
   * Example:
   * @code
   * cv::Mat display = rcv::vcat({first_image, second_image, third_image, fourth_image});
   * @endcode
   *
   * @param images A list of images to be concatenated in bottom-to-top order
   * @param fill A color to fill in the background of an image if it is smaller than the other
   */
  cv::Mat vcat(std::vector<cv::Mat> const & images, cv::Scalar fill = cv::Scalar(0))
  {
    if(images.empty()) return cv::Mat();
    if(images.size() == 1) return images[0];

    for(size_t i=1; i<images.size(); ++i)
      if(images[i].type() != images[0].type())
        throw std::runtime_error("In rcv::vcat: mismatched types between "
            "images[0] (" + rcv::type2string(images[0].type()) + ") "
            "and images[" + std::to_string(i) + "] (" + rcv::type2string(images[i].type()) + ")");

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
  /*! @} */

  /*! \defgroup Plotting Plotting Utilities 
       Very simple line plotting
      @{ */

  // ######################################################################
  class AutoScaleIndicator { };
  //! An indicator to tell plot() that a value should be automatically scaled
  AutoScaleIndicator autoscale;

  template<class Iterator, class MaxScaleValue> 
    auto get_max_value(Iterator const begin, Iterator const end, MaxScaleValue max_value __attribute__ ((unused))) -> 
    typename std::enable_if<std::is_same<MaxScaleValue, AutoScaleIndicator>::value, typename std::remove_reference<decltype(*begin)>::type>::type
    { return *std::max_element(begin, end); }

  template<class Iterator, class MaxScaleValue> 
    auto get_max_value(Iterator const begin  __attribute__ ((unused)), Iterator const end  __attribute__ ((unused)), MaxScaleValue max_value) -> 
    typename std::enable_if<!std::is_same<MaxScaleValue, AutoScaleIndicator>::value, typename std::remove_reference<decltype(*begin)>::type>::type
    { return max_value; }

  template<class Iterator, class MaxScaleValue> 
    auto get_min_value(Iterator const begin, Iterator const end, MaxScaleValue min_value __attribute__ ((unused))) -> 
    typename std::enable_if<std::is_same<MaxScaleValue, AutoScaleIndicator>::value, typename std::remove_reference<decltype(*begin)>::type>::type
    { return *std::min_element(begin, end); }

  template<class Iterator, class MaxScaleValue> 
    auto get_min_value(Iterator const begin  __attribute__ ((unused)), Iterator const end  __attribute__ ((unused)), MaxScaleValue min_value) -> 
    typename std::enable_if<!std::is_same<MaxScaleValue, AutoScaleIndicator>::value, typename std::remove_reference<decltype(*begin)>::type>::type
    { return min_value; }

  // ######################################################################
  //! Plot the values as a simple line plot
  /*! Both the minimum and maximum values can be autoscaled by passing the special rcv::autoscale value to them.
   *  @param begin An iterator pointing to the beginning of the data to plot
   *  @param end An iterator pointing to one past the end of the data to plot
   *  @param plot_size The size of the plot (in pixels), given as cv::Size(width,height)
   *  @param min_value The minimum value to plot, or rcv::autoscale to automatically scale this value 
   *  @param max_value The maximum value to plot, or rcv::autoscale to automatically scale this value
   *  @param line_color The color of the plot line 
   *  @param line_width The width of the plot line
   *  @param image_type The type of image to create 
   *  @param write_limits Show the minimum and maximum values on the plot */
  template<class Iterator, class MinScaleValue=AutoScaleIndicator, class MaxScaleValue=AutoScaleIndicator>
  cv::Mat plot(Iterator const begin, Iterator const end, cv::Size plot_size,
      MinScaleValue min_value = autoscale, MaxScaleValue max_value = autoscale,
      cv::Scalar line_color=cv::Scalar::all(255), int line_width=1, int image_type=CV_8UC3, bool write_limits = false)
  {
    typedef typename std::remove_reference<decltype(*begin)>::type T;

    cv::Mat plot = cv::Mat::zeros(plot_size, image_type);

    T min_value_ = get_min_value(begin, end, min_value);
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

    if(write_limits)
    {
      auto to_string = [](T value)
      {
        std::ostringstream out;
        out << std::setprecision(2) << std::fixed << value;
        return out.str();
      };

      int baseline;
      auto const font = CV_FONT_HERSHEY_PLAIN;
      float const font_scale = 1.0;
      cv::Size max_loc = cv::getTextSize(to_string(max_value_), font, font_scale, 1, &baseline);
      cv::putText(plot, to_string(max_value_), cv::Point(0, max_loc.height+5), font, font_scale, cv::Scalar::all(255));
      cv::putText(plot, to_string(min_value_), cv::Point(0, plot.rows-5), font, font_scale, cv::Scalar::all(255));
    }
    return plot; 
  }

  /*! @} */

  /*! \defgroup Colorizing Image Colorizing Utilities
      @{ */ 

  // ######################################################################
  //! Colorize the input using Dave Green's 'cubehelix' algorithm
  /*! This implementation based on Jim Davenport's python implementation found here: https://github.com/jradavenport/cubehelix/ 

    cubehelix contains a handy named parameter creation helper,
    @code
    cubehelix colorizer = cubehelix::create().start(.6).rot(1.5).nlev(128).reverse();

    cv::Mat grayscale;
    cv::Mat colorized = colorizer(grayscale);
    @endcode
   
   
   */
  class cubehelix
  {
    public:

      //! A helper struct to create cubehelix instances using named parameters
      /*! @code
          cubehelix colorizer = cubehelix::create().start(.6).rot(1.5).nlev(128).reverse();

          cv::Mat grayscale;
          cv::Mat colorized = colorizer(grayscale);
          @endcode */
      struct create
      {
        create() :
          nlev_(256), start_(0.5), rot_(-1.5), gamma_(1.0), hue_(1.2), reverse_(false) 
        { }

        operator cubehelix()
        {
          return cubehelix(nlev_, start_, rot_, gamma_, hue_, reverse_);
        }

        //! The number of color levels in the color map.
        create & nlev(size_t val)  { nlev_ = val; return *this; }

        //! The starting position in the color space. 0=blue, 1=red, 2=green. Defaults to 0.5.
        create & start(float val)  { start_ = val; return *this; }

        //! The number of rotations through the rainbow.
        /*! Can be positive or negative, indicating direction of rainbow.
            Negative values correspond to Blue->Red direction. */
        create & rot(float val)    { rot_ = val; return *this; }

        //! The gamma correction for intensity.
        create & gamma(float val)  { gamma_ = val; return *this; }

        //! The hue intensity factor.
        create & hue(float val)    { hue_ = val; return *this; }

        //! Set to True to reverse the color map.
        /*! Will go from black to white. Good for density plots where shade~density. */
        create & reverse()         { reverse_ = !reverse_; return *this; }

        size_t nlev_;
        float start_, rot_, gamma_, hue_;
        bool reverse_;
      };


      //! Construct a cubehelix object and initialize it's mapping tables
      /*! There are a lot of parameters here, so you can use the cubehelix::create class if you want named parameters
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
      }

   private:
      template<class T>
      cv::Mat process(cv::Mat const & input)
      {
        double minv, maxv;
        cv::minMaxLoc(input, &minv, &maxv);

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
  /*! @} */

}

/*!
 * \mainpage rcv:: Rand's OpenCV Utilities
 *
 * Please see the <a href="modules.html">Modules</a> page for the bulk of the documentation.
 
*/


#endif // RCV_HPP

