rcv::
=======
#### Randolph Charles Voorhies' OpenCV utilities

This is a set of general purpose OpenCV helpers that I have accumulated over the years. Here's an example of some of the library's functionality:

For more documentation, check out the [doxygen pages](http://randvoorhies.github.io/rcv).

```cpp
#include "rcv.hpp"

template<class T>
bool pixel_check(int r, int c, cv::Mat image, double param)
{
  assert(image.channels() == 1);
  return image.at<T>(r,c) < param;
}

int main()
{
  cv::Mat image0, image1, image2, image3;
  
  ////////////////// Image Concatenation //////////////////
  
  // Horizontally concatenate two images
  cv::Mat combo1 = rcv::hcat(image0, image1);
  
  // hcat is overloaded to work with vectors of images
  cv::Mat combo2 = rcv::hcat({image0, image1, image2, image3});
  
  // You can vertically concatenate as well
  cv::Mat combo3 = rcv::vcat({image0, image1, image2});
  
  // Creating grids of images is easy:
  cv::Mat grid = rcv::vcat(
                    rcv::hcat(image0, image1),
                    rcv::hcat(image2, image3)
                    );
                    
                
  ////////////////// Plotting //////////////////
  
  // There is some super basic plotting implemented:
  std::vector<float> values(100);
  
  // Plot with autoscale and a white line
  cv::Mat plot = rcv::plot(values.begin(), values.end(), cv::Size(300, 100));
  
  // Plot with a fixed scale (from 0 to 1) and a red line
  cv::Mat plot = rcv::plot(values.begin(), values.end(), cv::Size(300, 100),
                           0.0, 1.0, cv::Scalar(0,0,255));

  // Plot with an autoscaled minimum value, a fixed maximum value and a blue line
  cv::Mat plot = rcv::plot(values.begin(), values.end(), cv::Size(300, 100),
                           rcv::autoscale, 100.0, cv::Scalar(0,0,255));
  
  ////////////////// Color Mapping //////////////////
  
  // Dave Green's cubehelix algorithm is implemented to create beautiful RGB 
  // colormaps from single-channel data
  rcv::cubehelix maphelix;
  cv::Mat beautifulimage = maphelix(image0);

  ////////////////// Type Dispatching //////////////////

  cv::Mat mysterious_matrix = getMysteriousMatrix();
  bool pixel_ok = RCV_DISPATCH(mysterious_matrix.type(),
                               pixel_check, 0, 0, mysterious_matrix, 30.0);

  return 0;
}
```

_Beware that I really haven't optimized any of this code! The image concatentation is particuarly inefficient, as it involves copying a ton of data when concatenating many images. If you would like to speed this up, let me know as I have some ideas of how to do it._
