#include "modulated_deform_im2col_3d_cuda.h"
#include <cstdio>
#include <algorithm>
#include <cstring>

#define CUDA_KERNEL_LOOP(i, n)                          \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;   \
      i < (n);                                          \
      i += blockDim.x * gridDim.x)

const int CUDA_NUM_THREADS = 1024;
inline int GET_BLOCKS(const int N)
{
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}


__device__ float dmcn_im2col_bilinear(const float *bottom_data, const int data_width, const int data_depth,
                                      const int height, const int width, const int depth, float h, float w, float d)
{
  int h_low = floor(h);
  int w_low = floor(w);
  int d_low = floor(d);
  int h_high = h_low + 1;
  int w_high = w_low + 1;
  int d_high = d_low + 1;

  float lh = h - h_low;
  float lw = w - w_low;
  float ld = d - d_low;
  float hh = 1 - lh, hw = 1 - lw, hd = 1 - ld;

  float v1 = 0;
  if (h_low >= 0 && w_low >= 0 && d_low >= 0)
    v1 = bottom_data[h_low * data_width * data_depth + w_low + d_low];
  float v2 = 0;
  if (h_low >= 0 && w_high <= width - 1 && d_low >= 0)
    v2 = bottom_data[h_low * data_width * data_depth + w_high + d_low];
  float v3 = 0;
  if (h_high <= height - 1 && w_low >= 0 && d_low >= 0)
    v3 = bottom_data[h_high * data_width * data_depth + w_low + d_low];
  float v4 = 0;
  if (h_high <= height - 1 && w_high <= width - 1 && d_low >= 0)
    v4 = bottom_data[h_high * data_width * data_depth + w_high + d_low];
  if (h_low >= 0 && w_low >= 0 && d_high <= depth - 1)
    v1 = bottom_data[h_low * data_width * data_depth + w_low + d_high];
  if (h_low >= 0 && w_high <= width - 1 && d_high <= depth - 1)
    v2 = bottom_data[h_low * data_width * data_depth + w_high + d_high];
  if (h_high <= height - 1 && w_low >= 0 && d_high <= depth - 1)
    v3 = bottom_data[h_high * data_width * data_depth + w_low + d_high];
  if (h_high <= height - 1 && w_high <= width - 1 && d_high <= depth - 1)
    v4 = bottom_data[h_high * data_width * data_depth + w_high + d_high];

  float w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;
  float d1 = hh * hd, d2 = hh * ld, d3 = lh * hd, d4 = lh * ld;

  float val = (w1 * v1 * d1 + w2 * v2 * d2 + w3 * v3 * d3 + w4 * v4 * d4);
  return val;
}

__device__ float dmcn_get_gradient_weight(float argmax_h, float argmax_w, float argmax_d,
                                          const int h, const int w, const int d, const int height, const int width, const int depth)
{
  if (argmax_h <= -1 || argmax_h >= height || argmax_w <= -1 || argmax_w >= width || argmax_d <= -1 || argmax_d >= depth)
  {
    //empty
    return 0;
  }

  int argmax_h_low = floor(argmax_h);
  int argmax_w_low = floor(argmax_w);
  int argmax_d_low = floor(argmax_d);
  int argmax_h_high = argmax_h_low + 1;
  int argmax_w_high = argmax_w_low + 1;
  int argmax_d_high = argmax_d_low + 1;

  float weight = 0;
  if (h == argmax_h_low && w == argmax_w_low && d == argmax_d_low)
    weight = (h + 1 - argmax_h) * (w + 1 - argmax_w) * (d + 1 - argmax_d);
  if (h == argmax_h_low && w == argmax_w_high && d == argmax_d_low)
    weight = (h + 1 - argmax_h) * (argmax_w + 1 - w) * (d + 1 - argmax_d);
  if (h == argmax_h_high && w == argmax_w_low && d == argmax_d_low)
    weight = (argmax_h + 1 - h) * (w + 1 - argmax_w) * (d + 1 - argmax_d);
  if (h == argmax_h_high && w == argmax_w_high && d == argmax_d_low)
    weight = (argmax_h + 1 - h) * (argmax_w + 1 - w) * (d + 1 - argmax_d);
  if (h == argmax_h_low && w == argmax_w_low && d == argmax_d_high)
    weight = (h + 1 - argmax_h) * (w + 1 - argmax_w) * (argmax_d + 1 - d);
  if (h == argmax_h_low && w == argmax_w_high && d == argmax_d_high)
    weight = (h + 1 - argmax_h) * (argmax_w + 1 - w) * (argmax_d + 1 - d);
  if (h == argmax_h_high && w == argmax_w_low && d == argmax_d_high)
    weight = (argmax_h + 1 - h) * (w + 1 - argmax_w) * (argmax_d + 1 - d);
  if (h == argmax_h_high && w == argmax_w_high && d == argmax_d_high)
    weight = (argmax_h + 1 - h) * (argmax_w + 1 - w) * (argmax_d + 1 - d);
  return weight;
}

__device__ float dmcn_get_coordinate_weight(float argmax_h, float argmax_w, float argmax_d,
                                            const int height, const int width, const int depth, const float *im_data,
                                            const int data_width, const int data_depth, const int bp_dir)
{
  if (argmax_h <= -1 || argmax_h >= height || argmax_w <= -1 || argmax_w >= width || argmax_d <= -1 || argmax_d >= depth)
  {
    //empty
    return 0;
  }

  int argmax_h_low = floor(argmax_h);
  int argmax_w_low = floor(argmax_w);
  int argmax_d_low = floor(argmax_d);
  int argmax_h_high = argmax_h_low + 1;
  int argmax_w_high = argmax_w_low + 1;
  int argmax_d_high = argmax_d_low + 1;

  float weight = 0;

  if (bp_dir == 0) {
    if (argmax_h_low >= 0 && argmax_w_low >= 0 && argmax_d_low >= 0)
        weight += -1 * (argmax_w_low + 1 - argmax_w) * (argmax_d_low + 1 - argmax_d) * im_data[argmax_h_low * data_width * data_depth + argmax_w_low + argmax_d_low];
    if (argmax_h_low >= 0 && argmax_w_high <= width - 1 && argmax_d_low >= 0)
        weight += -1 * (argmax_w - argmax_w_low) * (argmax_d_low + 1 - argmax_d) * im_data[argmax_h_low * data_width * data_depth + argmax_w_high + argmax_d_low];
    if (argmax_h_high <= height - 1 && argmax_w_low >= 0 && argmax_d_low >= 0)
        weight += (argmax_w_low + 1 - argmax_w) * (argmax_d_low + 1 - argmax_d) * im_data[argmax_h_high * data_width * data_depth + argmax_w_low] + argmax_d_low;
    if (argmax_h_high <= height - 1 && argmax_w_high <= width - 1 && argmax_d_low >= 0)
        weight += (argmax_w - argmax_w_low) * (argmax_d_low + 1 - argmax_d) * im_data[argmax_h_high * data_width * data_depth + argmax_w_high + argmax_d_low];
    if (argmax_h_low >= 0 && argmax_w_low >= 0 && argmax_d_high <= depth - 1)
        weight += -1 * (argmax_w_low + 1 - argmax_w) * (argmax_d - argmax_d_low) * im_data[argmax_h_low * data_width * data_depth + argmax_w_low + argmax_d_high];
    if (argmax_h_low >= 0 && argmax_w_high <= width - 1 && argmax_d_high <= depth - 1)
        weight += -1 * (argmax_w - argmax_w_low) * (argmax_d - argmax_d_low) * im_data[argmax_h_low * data_width * data_depth + argmax_w_high + argmax_d_high];
    if (argmax_h_high <= height - 1 && argmax_w_low >= 0 && argmax_d_high <= depth - 1)
        weight += (argmax_w_low + 1 - argmax_w) * (argmax_d - argmax_d_low) * im_data[argmax_h_high * data_width * data_depth + argmax_w_low + argmax_d_high];
    if (argmax_h_high <= height - 1 && argmax_w_high <= width - 1 && argmax_d_high <= depth - 1)
        weight += (argmax_w - argmax_w_low) * (argmax_d - argmax_d_low) * im_data[argmax_h_high * data_width * data_depth + argmax_w_high + argmax_d_high];
  } else if (bp_dir == 1) {
    if (argmax_h_low >= 0 && argmax_w_low >= 0 && argmax_d_low >= 0)
        weight += -1 * (argmax_h_low + 1 - argmax_h) * im_data[argmax_h_low * data_width * data_depth + argmax_w_low + argmax_d_low];
    if (argmax_h_low >= 0 && argmax_w_high <= width - 1 && argmax_d_low >= 0)
        weight += (argmax_h_low + 1 - argmax_h) * im_data[argmax_h_low * data_width * data_depth + argmax_w_high + argmax_d_low];
    if (argmax_h_high <= height - 1 && argmax_w_low >= 0 && argmax_d_low >= 0)
        weight += -1 * (argmax_h - argmax_h_low) * im_data[argmax_h_high * data_width * data_depth + argmax_w_low + argmax_d_low];
    if (argmax_h_high <= height - 1 && argmax_w_high <= width - 1 && argmax_d_low >= 0)
        weight += (argmax_h - argmax_h_low) * im_data[argmax_h_high * data_width * data_depth + argmax_w_high + argmax_d_low];
    if (argmax_h_low >= 0 && argmax_w_low >= 0 && argmax_d_high <= depth - 1)
        weight += -1 * (argmax_h_low + 1 - argmax_h) * im_data[argmax_h_low * data_width * data_depth + argmax_w_low + argmax_d_high];
    if (argmax_h_low >= 0 && argmax_w_high <= width - 1 && argmax_d_high <= depth - 1)
        weight += (argmax_h_low + 1 - argmax_h) * im_data[argmax_h_low * data_width * data_depth + argmax_w_high + argmax_d_high];
    if (argmax_h_high <= height - 1 && argmax_w_low >= 0 && argmax_d_high <= depth - 1)
        weight += -1 * (argmax_h - argmax_h_low) * im_data[argmax_h_high * data_width * data_depth + argmax_w_low + argmax_d_high];
    if (argmax_h_high <= height - 1 && argmax_w_high <= width - 1 && argmax_d_high <= depth - 1)
        weight += (argmax_h - argmax_h_low) * im_data[argmax_h_high * data_width * data_depth + argmax_w_high + argmax_d_high];
  }

  return weight;
}

__global__ void modulated_deformable_im2col_gpu_kernel(const int n,
                                                       const float *data_im, const float *data_offset, const float *data_mask,
                                                       const int height, const int width, const int depth, const int kernel_h, const int kernel_w, const int kernel_d,
                                                       const int pad_h, const int pad_w, const int pad_d,
                                                       const int stride_h, const int stride_w, const int stride_d,
                                                       const int dilation_h, const int dilation_w, const int dilation_d,
                                                       const int channel_per_deformable_group,
                                                       const int batch_size, const int num_channels, const int deformable_group,
                                                       const int height_col, const int width_col, const int depth_col,
                                                       float *data_col)
{
  CUDA_KERNEL_LOOP(index, n)
  {
    // index index of output matrix
    const int d_col = index % depth_col;
    const int w_col = index % width_col;
    const int h_col = (index / depth_col / width_col) % height_col;
    const int b_col = (index / depth_col / width_col / height_col) % batch_size;
    const int c_im = (index / depth_col / width_col / height_col) / batch_size;
    const int c_col = c_im * kernel_h * kernel_w * kernel_d;

    // compute deformable group index
    const int deformable_group_index = c_im / channel_per_deformable_group;

    const int h_in = h_col * stride_h - pad_h;
    const int w_in = w_col * stride_w - pad_w;
    const int d_in = d_col * stride_d - pad_d;

    float *data_col_ptr = data_col + (((c_col * batch_size + b_col) * height_col + h_col) * width_col + w_col) * depth_col + d_col;
    //const float* data_im_ptr = data_im + ((b_col * num_channels + c_im) * height + h_in) * width + w_in;
    const float *data_im_ptr = data_im + (b_col * num_channels + c_im) * height * width * depth;
    const float *data_offset_ptr = data_offset + (b_col * deformable_group + deformable_group_index) * 2 * kernel_h * kernel_w * height_col * width_col;

    const float *data_mask_ptr = data_mask + (b_col * deformable_group + deformable_group_index) * kernel_h * kernel_w * height_col * width_col;

    for (int i = 0; i < kernel_h; ++i)
    {
      for (int j = 0; j < kernel_w; ++j)
      {
        for (int k = 0; k < kernel_d; ++k)
        {
            const int data_offset_h_ptr = (((3 * (i * kernel_w * kernel_d + j + k)) * height_col + h_col) * width_col + w_col) * depth_col + d_col;
            const int data_offset_w_ptr = (((3 * (i * kernel_w * kernel_d + j + k) + 1) * height_col + h_col) * width_col + w_col) * depth_col + d_col;
            const int data_offset_d_ptr = (((3 * (i * kernel_w * kernel_d + j + k) + 1) * height_col + h_col) * width_col + w_col) * depth_col + d_col;
            const float offset_h = data_offset_ptr[data_offset_h_ptr];
            const float offset_w = data_offset_ptr[data_offset_w_ptr];
            const float mask = data_mask_ptr[data_mask_hw_ptr];
            float val = static_cast<float>(0);
            const float h_im = h_in + i * dilation_h + offset_h;
            const float w_im = w_in + j * dilation_w + offset_w;
            const float d_im = d_in + j * dilation_d + offset_d;
            //if (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width) {
            if (h_im > -1 && w_im > -1 && d_im > -1 && h_im < height && w_im < width && d_im < depth)
            {
            //const float map_h = i * dilation_h + offset_h;
            //const float map_w = j * dilation_w + offset_w;
            //const int cur_height = height - h_in;
            //const int cur_width = width - w_in;
            //val = dmcn_im2col_bilinear(data_im_ptr, width, cur_height, cur_width, map_h, map_w);
            val = dmcn_im2col_bilinear(data_im_ptr, width, height, width, depth, h_im, w_im, d_im);
            }
            *data_col_ptr = val * mask;
            data_col_ptr += batch_size * height_col * width_col * depth_col;
            //data_col_ptr += height_col * width_col;
        }
      }
    }
  }
}

__global__ void modulated_deformable_col2im_gpu_kernel(const int n,
                                                       const float *data_col, const float *data_offset, const float *data_mask,
                                                       const int channels, const int height, const int width, const int depth,
                                                       const int kernel_h, const int kernel_w, const int kernel_d,
                                                       const int pad_h, const int pad_w, const int pad_d,
                                                       const int stride_h, const int stride_w, const int stride_d,
                                                       const int dilation_h, const int dilation_w, const int dilation_d,
                                                       const int channel_per_deformable_group,
                                                       const int batch_size, const int deformable_group,
                                                       const int height_col, const int width_col, const int depth_col,
                                                       float *grad_im)
{
  CUDA_KERNEL_LOOP(index, n)
  {
    const int k = (index / depth_col / width_col / height_col / batch_size) % kernel_d;
    const int j = (index / depth_col / width_col / height_col / batch_size) % kernel_w;
    const int i = (index / depth_col / width_col / height_col / batch_size / kernel_w) % kernel_h;
    const int c = index / depth_col / width_col / height_col / batch_size / kernel_d / kernel_w / kernel_h;
    // compute the start and end of the output

    const int deformable_group_index = c / channel_per_deformable_group;

    int d_out = index % depth_col;
    int w_out = index % width_col;
    int h_out = (index / depth_col / width_col) % height_col;
    int b = (index / depth_col / width_col / height_col) % batch_size;
    int d_in = d_out * stride_d - pad_d;
    int w_in = w_out * stride_w - pad_w;
    int h_in = h_out * stride_h - pad_h;

    const float *data_offset_ptr = data_offset + (b * deformable_group + deformable_group_index) * 2 * kernel_h * kernel_w * height_col * width_col;
    const float *data_mask_ptr = data_mask + (b * deformable_group + deformable_group_index) * kernel_h * kernel_w * height_col * width_col;
    const int data_offset_h_ptr = (((3 * (i * kernel_w * kernel_d + j + k)) * height_col + h_out) * width_col + w_out) * depth_col + d_out;
    const int data_offset_w_ptr = (((3 * (i * kernel_w * kernel_d + j + k) + 1) * height_col + h_out) * width_col + w_out) * depth_col + d_out;
    const int data_offset_d_ptr = (((3 * (i * kernel_w * kernel_d + j + k) + 1) * height_col + h_out) * width_col + w_out) * depth_col + d_out;
    const int data_mask_hw_ptr = (((i * kernel_w + k * kernel_d + j) * height_col + h_out) * width_col + w_out) * depth_col + d_out;
    const float offset_h = data_offset_ptr[data_offset_h_ptr];
    const float offset_w = data_offset_ptr[data_offset_w_ptr];
    const float offset_d = data_offset_ptr[data_offset_d_ptr];
    const float mask = data_mask_ptr[data_mask_hw_ptr];
    const float cur_inv_h_data = h_in + i * dilation_h + offset_h;
    const float cur_inv_w_data = w_in + j * dilation_w + offset_w;
    const float cur_inv_d_data = d_in + k * dilation_d + offset_d;

    const float cur_top_grad = data_col[index] * mask;
    const int cur_h = (int)cur_inv_h_data;
    const int cur_w = (int)cur_inv_w_data;
    const int cur_d = (int)cur_inv_d_data;
    for (int dy = -2; dy <= 2; dy++)
    {
      for (int dx = -2; dx <= 2; dx++)
      {
        for (int dz = -2; dz <=2; dz++)
        {
            if (cur_h + dy >= 0 && cur_h + dy < height &&
                cur_w + dx >= 0 && cur_w + dx < width &&
                cur_d + dz >= 0 && cur_d + dz < depth &&
                abs(cur_inv_h_data - (cur_h + dy)) < 1 &&
                abs(cur_inv_w_data - (cur_w + dx)) < 1 &&
                abs(cur_inv_d_data - (cur_d + dz)) < 1)
            {
            int cur_bottom_grad_pos = (((b * channels + c) * height + cur_h + dy) * width + cur_w + dx) * depth + cur_d + dz;
            float weight = dmcn_get_gradient_weight(cur_inv_h_data, cur_inv_w_data, cur_inv_d_data, cur_h + dy, cur_w + dx, cur_d + dz, height, width, depth);
            atomicAdd(grad_im + cur_bottom_grad_pos, weight * cur_top_grad);
            }
        }
      }
    }
  }
}

__global__ void modulated_deformable_col2im_coord_gpu_kernel(const int n,
                                                             const float *data_col, const float *data_im,
                                                             const float *data_offset, const float *data_mask,
                                                             const int channels, const int height, const int width, const int depth,
                                                             const int kernel_h, const int kernel_w, const int kernel_d,
                                                             const int pad_h, const int pad_w, const int pad_d,
                                                             const int stride_h, const int stride_w, const int stride_d,
                                                             const int dilation_h, const int dilation_w, const int dilation_d,
                                                             const int channel_per_deformable_group,
                                                             const int batch_size, const int offset_channels, const int deformable_group,
                                                             const int height_col, const int width_col, const int depth_col,
                                                             float *grad_offset, float *grad_mask)
{
  CUDA_KERNEL_LOOP(index, n)
  {
    float val = 0, mval = 0;
    int d = index % depth_col;
    int w = index % width_col;
    int h = (index / depth_col / width_col) % height_col;
    int c = (index / depth_col / width_col / height_col) % offset_channels;
    int b = (index / depth_col / width_col / height_col) / offset_channels;
    // compute the start and end of the output

    const int deformable_group_index = c / (2 * kernel_h * kernel_w);
    const int col_step = kernel_h * kernel_w * kernel_d;
    int cnt = 0;
    const float *data_col_ptr = data_col + deformable_group_index * channel_per_deformable_group * batch_size * width_col * height_col * depth_col;
    const float *data_im_ptr = data_im + (b * deformable_group + deformable_group_index) * channel_per_deformable_group / kernel_h / kernel_w / kernel_d * height * width * depth;
    const float *data_offset_ptr = data_offset + (b * deformable_group + deformable_group_index) * 3 * kernel_h * kernel_w * kernel_d * height_col * width_col * depth_col;
    const float *data_mask_ptr = data_mask + (b * deformable_group + deformable_group_index) * kernel_h * kernel_w * kernel_d * height_col * width_col * depth_col;

    const int offset_c = c - deformable_group_index * 3 * kernel_h * kernel_w * kernel_d;

    for (int col_c = (offset_c / 2); col_c < channel_per_deformable_group; col_c += col_step)
    {
      const int col_pos = ((((col_c * batch_size + b) * height_col) + h) * width_col + w) * depth_col + d;
      const int bp_dir = offset_c % 2;

      int k = (col_pos / depth_col / height_col / batch_size) % kernel_d;
      int j = (col_pos / width_col / height_col / batch_size) % kernel_w;
      int i = (col_pos / depth_col / width_col / height_col / batch_size / kernel_d / kernel_w) % kernel_h;
      int d_out = col_pos % depth_col;
      int w_out = col_pos % width_col;
      int h_out = (col_pos / depth_col / width_col) % height_col;
      int d_in = d_out * stride_d - pad_d;
      int w_in = w_out * stride_w - pad_w;
      int h_in = h_out * stride_h - pad_h;
      const int data_offset_h_ptr = (((3 * (i * kernel_w * kernel_d + j + k)) * height_col + h_out) * width_col + w_out) * depth_col + d_out;
      const int data_offset_w_ptr = (((3 * (i * kernel_w * kernel_d + j + k) + 1) * height_col + h_out) * width_col + w_out) * depth_col + d_out;
      const int data_offset_d_ptr = (((3 * (i * kernel_w * kernel_d + j + k) + 1) * height_col + h_out) * width_col + w_out) * depth_col + d_out;
      const int data_mask_hw_ptr = (((i * kernel_w + k * kernel_d + j) * height_col + h_out) * width_col + w_out) * depth_col + d_out;
      const float offset_h = data_offset_ptr[data_offset_h_ptr];
      const float offset_w = data_offset_ptr[data_offset_w_ptr];
      const float offset_d = data_offset_ptr[data_offset_d_ptr];
      const float mask = data_mask_ptr[data_mask_hw_ptr];
      float inv_h = h_in + i * dilation_h + offset_h;
      float inv_w = w_in + j * dilation_w + offset_w;
      float inv_d = d_in + k * dilation_d + offset_d;
      if (inv_h <= -1 || inv_w <= -1 || inv_d <= -1 || inv_h >= height || inv_w >= width || inv_d >= depth)
      {
        inv_h = inv_w = inv_d = -2;
      }
      else
      {
        mval += data_col_ptr[col_pos] * dmcn_im2col_bilinear(data_im_ptr + cnt * height * width * depth, width, height, depth, width * depth, inv_h, inv_w, inv_d);
      }
      const float weight = dmcn_get_coordinate_weight(
          inv_h, inv_w, inv_d,
          height, width, depth, data_im_ptr + cnt * height * width * depth, width * depth, bp_dir);
      val += weight * data_col_ptr[col_pos] * mask;
      cnt += 1;
    }
    // KERNEL_ASSIGN(grad_offset[index], offset_req, val);
    grad_offset[index] = val;
    if (offset_c % 2 == 0)
      // KERNEL_ASSIGN(grad_mask[(((b * deformable_group + deformable_group_index) * kernel_h * kernel_w + offset_c / 2) * height_col + h) * width_col + w], mask_req, mval);
      grad_mask[((((b * deformable_group + deformable_group_index) * kernel_h * kernel_w * kernel_d + offset_c / 3) * height_col + h) * width_col + w) * depth_col + d] = mval;
  }
}

void modulated_deformable_im2col_cuda(cudaStream_t stream,
  const float* data_im, const float* data_offset, const float* data_mask,
  const int batch_size, const int channels, const int height_im, const int width_im, const int depth_im,
  const int height_col, const int width_col, const int depth_col, const int kernel_h, const int kernel_w, const int kernel_d,
  const int pad_h, const int pad_w, const int pad_d, const int stride_h, const int stride_w, const int stride_d, 
  const int dilation_h, const int dilation_w, const int dilation_d,
  const int deformable_group, float* data_col) {
  // num_axes should be smaller than block size
  const int channel_per_deformable_group = channels / deformable_group;
  const int num_kernels = channels * batch_size * height_col * width_col * depth_col;
  modulated_deformable_im2col_gpu_kernel
      <<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS,
          0, stream>>>(
      num_kernels, data_im, data_offset, data_mask, height_im, width_im, depth_im, kernel_h, kernel_w, kernel_d,
      pad_h, pad_w, pad_d, stride_h, stride_w, stride_d, dilation_h, dilation_w, dilation_d, channel_per_deformable_group,
      batch_size, channels, deformable_group, height_col, width_col, depth_col, data_col);
  
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("error in modulated_deformable_im2col_cuda: %s\n", cudaGetErrorString(err));
  }

}

void modulated_deformable_col2im_cuda(cudaStream_t stream,
  const float* data_col, const float* data_offset, const float* data_mask,
  const int batch_size, const int channels, const int height_im, const int width_im, const int depth_im,
  const int height_col, const int width_col, const int depth_col, const int kernel_h, const int kernel_w, const int kernel_d,
  const int pad_h, const int pad_w, const int pad_d, const int stride_h, const int stride_w, const int stride_d, 
  const int dilation_h, const int dilation_w, const int dilation_d,
  const int deformable_group, float* grad_im){

  const int channel_per_deformable_group = channels / deformable_group;
  const int num_kernels = channels * kernel_h * kernel_w * kernel_d * batch_size * height_col * width_col * depth_col;
  modulated_deformable_col2im_gpu_kernel
      <<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS,
          0, stream>>>(
        num_kernels, data_col, data_offset, data_mask, channels, height_im, width_im, depth_im,
        kernel_h, kernel_w, kernel_d, pad_h, pad_h, pad_d, stride_h, stride_w, stride_d,
        dilation_h, dilation_w, dilation_d, channel_per_deformable_group,
        batch_size, deformable_group, height_col, width_col, depth_col, grad_im);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("error in modulated_deformable_col2im_cuda: %s\n", cudaGetErrorString(err));
  }

}

void modulated_deformable_col2im_coord_cuda(cudaStream_t stream,
  const float* data_col, const float* data_im, const float* data_offset, const float* data_mask,
  const int batch_size, const int channels, const int height_im, const int width_im, 
  const int height_col, const int width_col, const int depth_col, const int kernel_h, const int kernel_w, const int kernel_d,
  const int pad_h, const int pad_w, const int pad_d, const int stride_h, const int stride_w, const int stride_d,
  const int dilation_h, const int dilation_w, const int dilation_d,
  const int deformable_group,
  float* grad_offset, float* grad_mask) {
  const int num_kernels = batch_size * height_col * width_col * depth_col * 3 * kernel_h * kernel_w * kernel_d * deformable_group;
  const int channel_per_deformable_group = channels * kernel_h * kernel_w * kernel_d / deformable_group;
  modulated_deformable_col2im_coord_gpu_kernel
      <<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS,
        0, stream>>>(
        num_kernels, data_col, data_im, data_offset, data_mask, channels, height_im, width_im, depth_im,
        kernel_h, kernel_w, kernel_d, pad_h, pad_w, pad_d, stride_h, stride_w, stride_d,
        dilation_h, dilation_w, dilation_d, channel_per_deformable_group,
        batch_size, 3 * kernel_h * kernel_w * kernel_d * deformable_group, deformable_group, height_col, width_col, depth_col,
        grad_offset, grad_mask);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("error in modulated_deformable_col2im_coord_cuda: %s\n", cudaGetErrorString(err));
  }
}