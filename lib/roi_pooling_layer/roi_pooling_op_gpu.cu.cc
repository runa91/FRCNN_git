#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include <stdio.h>
#include <cfloat>
#include <iostream>
#include <math.h>

#include "roi_pooling_op_gpu.h"

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

using std::max;
using std::min;

// namespace tensorflow {
using namespace tensorflow;

template <typename Dtype>
__global__ void ROIPoolForward(const int nthreads, const Dtype* bottom_data,
    const Dtype spatial_scale, const int height, const int width, 
    const int channels, const int pooled_height, const int pooled_width,
    const Dtype* bottom_rois,  const Dtype* bottom_orientations, Dtype* top_data, int* argmax_data)
{
  CUDA_1D_KERNEL_LOOP(index, nthreads) 
  {
    // (n, ph, pw, c) is an element in the pooled output
    // nthreads = outputsize = num_rois * pooled_height * pooled_width * channels;
    int n = index;
    int c = n % channels;
    n /= channels;
    int pw = n % pooled_width;
    n /= pooled_width;
    int ph = n % pooled_height;
    n /= pooled_height;

    bottom_rois += n * 5;
    int roi_batch_ind = bottom_rois[0];
    int roi_start_w = round(bottom_rois[1] * spatial_scale);
    int roi_start_h = round(bottom_rois[2] * spatial_scale);
    int roi_end_w = round(bottom_rois[3] * spatial_scale);
    int roi_end_h = round(bottom_rois[4] * spatial_scale);

    bottom_orientations += n * 1;
    float angle = - bottom_orientations[0];           // is float ok ??  // what sign?
    float sin_a = sin(angle);
    float cos_a = cos(angle);

    // Force malformed ROIs to be 1x1
    int roi_width = max(roi_end_w - roi_start_w + 1, 1);
    int roi_height = max(roi_end_h - roi_start_h + 1, 1);
    Dtype bin_size_h = static_cast<Dtype>(roi_height)
                       / static_cast<Dtype>(pooled_height);
    Dtype bin_size_w = static_cast<Dtype>(roi_width)
                       / static_cast<Dtype>(pooled_width);

    int hstart = static_cast<int>(floor(static_cast<Dtype>(ph)
                                        * bin_size_h));
    int wstart = static_cast<int>(floor(static_cast<Dtype>(pw)
                                        * bin_size_w));
    int hend = static_cast<int>(ceil(static_cast<Dtype>(ph + 1)
                                     * bin_size_h));
    int wend = static_cast<int>(ceil(static_cast<Dtype>(pw + 1)
                                     * bin_size_w));

    // Add roi offsets and clip to input boundaries
    hstart = min(max(hstart + roi_start_h, 0), height);
    hend = min(max(hend + roi_start_h, 0), height);
    wstart = min(max(wstart + roi_start_w, 0), width);
    wend = min(max(wend + roi_start_w, 0), width);
    bool is_empty = (hend <= hstart) || (wend <= wstart);



    // Define an empty pooling region to be zero
    Dtype maxval = is_empty ? 0 : -FLT_MAX;
    // If nothing is pooled, argmax = -1 causes nothing to be backprop'd
    int maxidx = -1;

    // define the point around which we rotate -> center of the rectangle
    Dtype roi_mid_h = (static_cast<Dtype>(roi_start_h) + static_cast<Dtype>(roi_end_h))/2.0;
    Dtype roi_mid_w = (static_cast<Dtype>(roi_start_w) + static_cast<Dtype>(roi_end_w))/2.0;
    Dtype h_from_mid;
    Dtype w_from_mid;
    Dtype h_from_mid_rot;
    Dtype w_from_mid_rot;
    int h_rot;
    int w_rot;

    bottom_data += roi_batch_ind * channels * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        w_from_mid = static_cast<Dtype>(w) - roi_mid_w;
        h_from_mid = static_cast<Dtype>(h) - roi_mid_h;
        w_from_mid_rot = static_cast<Dtype>(cos_a)*w_from_mid - static_cast<Dtype>(sin_a)*h_from_mid;   // check direction of rotation: https://ch.mathworks.com/help/phased/ref/rotz.html?requestedDomain=www.mathworks.com
        h_from_mid_rot = static_cast<Dtype>(sin_a)*w_from_mid + static_cast<Dtype>(cos_a)*h_from_mid;

        w_rot = static_cast<int>(round(w_from_mid_rot + roi_mid_w));
        h_rot = static_cast<int>(round(h_from_mid_rot + roi_mid_h));

        //int bottom_index = (h * width + w) * channels + c;
        h_rot = min(max(h_rot, 0), height);     // check, if the point is still part of the image
        w_rot = min(max(w_rot, 0), width);
        int bottom_index = (h_rot * width + w_rot) * channels + c;

        if (bottom_data[bottom_index] > maxval) {
          maxval = bottom_data[bottom_index];
          maxidx = bottom_index;
        }
      }
    }
    top_data[index] = maxval;
    if (argmax_data != nullptr)         // check this (didn't change it with respect to the original version)
      argmax_data[index] = maxidx;      // save the knowledge about where the max element providing the output of the current cell of the current roi is located

    /*
    // Define an empty pooling region to be zero
    Dtype maxval = is_empty ? 0 : -FLT_MAX;
    // If nothing is pooled, argmax = -1 causes nothing to be backprop'd
    int maxidx = -1;
    bottom_data += roi_batch_ind * channels * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        int bottom_index = (h * width + w) * channels + c;
        if (bottom_data[bottom_index] > maxval) {
          maxval = bottom_data[bottom_index];
          maxidx = bottom_index;
        }
      }
    }
    top_data[index] = maxval;
    if (argmax_data != nullptr)
      argmax_data[index] = maxidx;      // save the knowledge about where the max element providing the output of the current cell of the current roi is located
    */
  }
}

bool ROIPoolForwardLaucher(
    const float* bottom_data, const float spatial_scale, const int num_rois, const int height,
    const int width, const int channels, const int pooled_height,
    const int pooled_width, const float* bottom_rois, const float* bottom_orientations,
    float* top_data, int* argmax_data, const Eigen::GpuDevice& d) 
{
  const int kThreadsPerBlock = 1024;
  const int output_size = num_rois * pooled_height * pooled_width * channels;
  cudaError_t err;

  ROIPoolForward<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock,
                       kThreadsPerBlock, 0, d.stream()>>>(
      output_size, bottom_data, spatial_scale, height, width, channels, pooled_height,
      pooled_width, bottom_rois, bottom_orientations, top_data, argmax_data);

  err = cudaGetLastError();
  if(cudaSuccess != err)
  {
    fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
    exit( -1 );
  }

  //std::cout << " ..................... roi pooling is still working (C++)........................";

  //std::cout << bottom_rois.dim(0);

  return d.ok();
}


template <typename Dtype>
__global__ void ROIPoolBackward(const int nthreads, const Dtype* top_diff,
    const int* argmax_data, const int num_rois, const Dtype spatial_scale,
    const int height, const int width, const int channels, 
    const int pooled_height, const int pooled_width, Dtype* bottom_diff,
    const Dtype* bottom_rois, const Dtype* bottom_orientations) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) 
  {
    // (n, h, w, c) coords in bottom data  -> this describes the VGG output data shape ...
    int n = index;
    int c = n % channels;
    n /= channels;
    int w = n % width;
    n /= width;
    int h = n % height;
    n /= height;

    Dtype gradient = 0;
    // Accumulate gradient over all ROIs that pooled this element
    for (int roi_n = 0; roi_n < num_rois; ++roi_n) 
    {
      const Dtype* offset_bottom_rois = bottom_rois + roi_n * 5;
      int roi_batch_ind = offset_bottom_rois[0];
      // Skip if ROI's batch index doesn't match n
      if (n != roi_batch_ind) {
        continue;
      }

      int roi_start_w = round(offset_bottom_rois[1] * spatial_scale);
      int roi_start_h = round(offset_bottom_rois[2] * spatial_scale);
      int roi_end_w = round(offset_bottom_rois[3] * spatial_scale);
      int roi_end_h = round(offset_bottom_rois[4] * spatial_scale);

      // define the center of the rectangle (point around which we rotate)
      Dtype roi_mid_h = (static_cast<Dtype>(roi_start_h) + static_cast<Dtype>(roi_end_h))/2.0;
      Dtype roi_mid_w = (static_cast<Dtype>(roi_start_w) + static_cast<Dtype>(roi_end_w))/2.0;
      Dtype roi_big_max_dist = sqrt(pow((static_cast<Dtype>(roi_end_h)-static_cast<Dtype>(roi_start_h)), 2.0) + pow((static_cast<Dtype>(roi_end_w)-static_cast<Dtype>(roi_start_w)), 2.0));
      int roi_start_w_big = static_cast<int>(floor(roi_mid_w - 0.5*roi_big_max_dist));
      int roi_end_w_big = static_cast<int>(ceil(roi_mid_w + 0.5*roi_big_max_dist));
      int roi_start_h_big = static_cast<int>(floor(roi_mid_h - 0.5*roi_big_max_dist));
      int roi_end_h_big = static_cast<int>(ceil(roi_mid_h + 0.5*roi_big_max_dist));

      /*const Dtype* offset_bottom_orientations = bottom_orientations + roi_n * 1;
      float angle = - static_cast<float>(offset_bottom_orientations[0]);           // is float fine ??
      float sin_a = sin(angle);
      float cos_a = cos(angle);*/

      // Skip if ROI doesn't include (h, w)     -> check if the 'pixel' we are looking at lays within the current roi
      /*const bool in_roi = (w >= roi_start_w && w <= roi_end_w &&
                           h >= roi_start_h && h <= roi_end_h);*/
      const bool in_roi = (w >= roi_start_w_big && w <= roi_end_w_big &&
                           h >= roi_start_h_big && h <= roi_end_h_big);     // h and w are the coordinates of the 'pixel' we are looking at
      if (!in_roi) {
        continue;
      }

      int offset = roi_n * pooled_height * pooled_width * channels;     // index of the first element of that roi (one roi has pooled_height * pooled_width * channels values)
      const Dtype* offset_top_diff = top_diff + offset;     // top diff is a pointer to the first element of 'out_backprop' = grads ...?
      // -> offset_top_dif is a pointer to the first element of the backpropagated gradients (top) which belong to the roi that we are currently looking at ...?
      const int* offset_argmax_data = argmax_data + offset;     // -> a pointer to the first element of argmax_data which belongs to the current roi

      // Compute feasible set of pooled units that could have pooled
      // this bottom unit

      // Force malformed ROIs to be 1x1
      int roi_width = max(roi_end_w - roi_start_w + 1, 1);
      int roi_height = max(roi_end_h - roi_start_h + 1, 1);

      /*Dtype bin_size_h = static_cast<Dtype>(roi_height)
                         / static_cast<Dtype>(pooled_height);
      Dtype bin_size_w = static_cast<Dtype>(roi_width)
                         / static_cast<Dtype>(pooled_width);*/


      // check for each cell of the pooled output (7*7), if it comes from 'pixel' at position h, w
      // this is not computationally efficient but a secure way to not miss any element -> think about how to make it more efficient...
      int phstart = 0;   //floor(static_cast<Dtype>(h - roi_start_h) / bin_size_h);
      int phend = pooled_height;    //ceil(static_cast<Dtype>(h - roi_start_h + 1) / bin_size_h);
      int pwstart = 0;   //floor(static_cast<Dtype>(w - roi_start_w) / bin_size_w);
      int pwend = pooled_width;  //bin_ceil(static_cast<Dtype>(w - roi_start_w + 1) / bin_size_w);

      phstart = min(max(phstart, 0), pooled_height);
      phend = min(max(phend, 0), pooled_height);
      pwstart = min(max(pwstart, 0), pooled_width);
      pwend = min(max(pwend, 0), pooled_width);

      for (int ph = phstart; ph < phend; ++ph) {
        for (int pw = pwstart; pw < pwend; ++pw) {
          if (offset_argmax_data[(ph * pooled_width + pw) * channels + c] == (h * width + w) * channels + c) 
          {
            gradient += offset_top_diff[(ph * pooled_width + pw) * channels + c];
          }
        }
      }
      /*int phstart = floor(static_cast<Dtype>(h - roi_start_h) / bin_size_h);
      int phend = ceil(static_cast<Dtype>(h - roi_start_h + 1) / bin_size_h);
      int pwstart = floor(static_cast<Dtype>(w - roi_start_w) / bin_size_w);
      int pwend = ceil(static_cast<Dtype>(w - roi_start_w + 1) / bin_size_w);

      phstart = min(max(phstart, 0), pooled_height);
      phend = min(max(phend, 0), pooled_height);
      pwstart = min(max(pwstart, 0), pooled_width);
      pwend = min(max(pwend, 0), pooled_width);

      for (int ph = phstart; ph < phend; ++ph) {
        for (int pw = pwstart; pw < pwend; ++pw) {
          if (offset_argmax_data[(ph * pooled_width + pw) * channels + c] == (h * width + w) * channels + c)
          {
            gradient += offset_top_diff[(ph * pooled_width + pw) * channels + c];
          }
        }
      }*/
    }
    bottom_diff[index] = gradient;      // bottom diff[index] contains the result / gradient for the 'pixel' we are currently looking at
  }
}


bool ROIPoolBackwardLaucher(const float* top_diff, const float spatial_scale, const int batch_size, const int num_rois,
    const int height, const int width, const int channels, const int pooled_height,
    const int pooled_width, const float* bottom_rois, const float* bottom_orientations,
    float* bottom_diff, const int* argmax_data, const Eigen::GpuDevice& d) 
{
  const int kThreadsPerBlock = 1024;
  const int output_size = batch_size * height * width * channels;
  cudaError_t err;

  ROIPoolBackward<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock,
                       kThreadsPerBlock, 0, d.stream()>>>(
      output_size, top_diff, argmax_data, num_rois, spatial_scale, height, width, channels, pooled_height,
      pooled_width, bottom_diff, bottom_rois, bottom_orientations);

  err = cudaGetLastError();
  if(cudaSuccess != err)
  {
    fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
    exit( -1 );
  }

  return d.ok();
}

// }  // namespace tensorflow

#endif  // GOOGLE_CUDA
