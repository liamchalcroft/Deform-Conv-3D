/*!
 * Copyright (c) 2017 Microsoft
 * Licensed under The MIT License [see LICENSE for details]
 * \file deformable_psroi_pooling.cu
 * \brief
 * \author Yi Li, Guodong Zhang, Jifeng Dai
*/
/***************** Adapted by Charles Shang *********************/
#include "deform_psroi_pooling_3d_cuda.h"
#include <cstdio>
#include <algorithm>
#include <cstring>

#define CUDA_KERNEL_LOOP(i, n)                        \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n);                                       \
       i += blockDim.x * gridDim.x)

const int CUDA_NUM_THREADS = 1024;
inline int GET_BLOCKS(const int N)
{
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

__device__ float bilinear_interp(
    const float *data,
    const float x,
    const float y,
    const float z,
    const int width,
    const int height,
    const int depth)
{
  int x1 = floor(x);
  int x2 = ceil(x);
  int y1 = floor(y);
  int y2 = ceil(y);
  int z1 = floor(z);
  int z2 = ceil(z);
  float dist_x = (float)(x - x1);
  float dist_y = (float)(y - y1);
  float dist_z = (float)(z - z1);
  float value111 = data[y1 * width + x1 * z1 * depth];
  float value121 = data[y2 * width + x1 * z1 * depth];
  float value211 = data[y1 * width + x2 * z1 * depth];
  float value221 = data[y2 * width + x2 * z1 * depth];
  float value112 = data[y1 * width + x1 * z2 * depth];
  float value122 = data[y2 * width + x1 * z2 * depth];
  float value212 = data[y1 * width + x2 * z2 * depth];
  float value222 = data[y2 * width + x2 * z2 * depth];
  float value = (1 - dist_x) * (1 - dist_y) * (1 - dist_z) * value111 + (1 - dist_x) * dist_y * (1 - dist_z) * value121 + dist_x * (1 - dist_y) * (1 - dist_z) * value211 + dist_x * dist_y * (1 - dist_z) * value221 + (1 - dist_x) * (1 - dist_y) * dist_z * value112 + (1 - dist_x) * dist_y * dist_z * value122 + dist_x * (1 - dist_y) * dist_z * value212 + dist_x * dist_y * dist_z * value222;
  return value;
}

__global__ void DeformablePSROIPoolForwardKernel(
    const int count,
    const float *bottom_data,
    const float spatial_scale,
    const int channels,
    const int height, const int width, const int depth,
    const int pooled_height, const int pooled_width, const int pooled_depth,
    const float *bottom_rois, const float *bottom_trans,
    const int no_trans,
    const float trans_std,
    const int sample_per_part,
    const int output_dim,
    const int group_size,
    const int part_size,
    const int num_classes,
    const int channels_each_class,
    float *top_data,
    float *top_count)
{
  CUDA_KERNEL_LOOP(index, count)
  {
    // The output is in order (n, ctop, ph, pw, pd)
    int pd = index % pooled_depth;
    int pw = index % pooled_width;
    int ph = (index / pooled_depth / pooled_width) % pooled_height;
    int ctop = (index / pooled_depth / pooled_width / pooled_height) % output_dim;
    int n = index / pooled_depth / pooled_width / pooled_height / output_dim;

    // [start, end) interval for spatial sampling
    const float *offset_bottom_rois = bottom_rois + n * 7;
    int roi_batch_ind = offset_bottom_rois[0];
    float roi_start_d = (float)(round(offset_bottom_rois[1])) * spatial_scale - 0.5;
    float roi_start_w = (float)(round(offset_bottom_rois[2])) * spatial_scale - 0.5;
    float roi_start_h = (float)(round(offset_bottom_rois[3])) * spatial_scale - 0.5;
    float roi_end_d = (float)(round(offset_bottom_rois[4]) + 1.) * spatial_scale - 0.5;
    float roi_end_w = (float)(round(offset_bottom_rois[4]) + 1.) * spatial_scale - 0.5;
    float roi_end_h = (float)(round(offset_bottom_rois[5]) + 1.) * spatial_scale - 0.5;

    // Force too small ROIs to be 1x1
    float roi_depth = max(roi_end_d - roi_start_d, 0.1);
    float roi_width = max(roi_end_w - roi_start_w, 0.1); //avoid 0
    float roi_height = max(roi_end_h - roi_start_h, 0.1);

    // Compute w and h at bottom
    float bin_size_h = roi_height / (float)(pooled_height);
    float bin_size_w = roi_width / (float)(pooled_width);
    float bin_size_d = roi_depth / (float)(pooled_depth);

    float sub_bin_size_h = bin_size_h / (float)(sample_per_part);
    float sub_bin_size_w = bin_size_w / (float)(sample_per_part);
    float sub_bin_size_d = bin_size_d / (float)(sample_per_part);

    int part_h = floor((float)(ph) / pooled_height * part_size);
    int part_w = floor((float)(pw) / pooled_width * part_size);
    int part_d = floor((float)(pd) / pooled_depth * part_size);
    int class_id = ctop / channels_each_class;
    float trans_x = no_trans ? (float)(0) : bottom_trans[((((n * num_classes + class_id) * 3) * part_size + part_h) * part_size + part_w) * part_size + part_d] * trans_std;
    float trans_y = no_trans ? (float)(0) : bottom_trans[((((n * num_classes + class_id) * 3 + 1) * part_size + part_h) * part_size + part_w) * part_size + part_d] * trans_std;
    float trans_z = no_trans ? (float)(0) : bottom_trans[((((n * num_classes + class_id) * 3) * part_size + part_h) * part_size + part_w) * part_size + part_d] * trans_std;

    float dstart = (float)(pd)*bin_size_d + roi_start_d;
    dstart += trans_z * roi_depth;
    float wstart = (float)(pw)*bin_size_w + roi_start_w;
    wstart += trans_x * roi_width;
    float hstart = (float)(ph)*bin_size_h + roi_start_h;
    hstart += trans_y * roi_height;

    float sum = 0;
    int count = 0;
    int gd = floor((float)(pd)*group_size / pooled_depth);
    int gw = floor((float)(pw)*group_size / pooled_width);
    int gh = floor((float)(ph)*group_size / pooled_height);
    gd = min(max(gd, 0), group_size - 1);
    gw = min(max(gw, 0), group_size - 1);
    gh = min(max(gh, 0), group_size - 1);

    const float *offset_bottom_data = bottom_data + (roi_batch_ind * channels) * height * width * depth;
    for (int ih = 0; ih < sample_per_part; ih++)
    {
      for (int iw = 0; iw < sample_per_part; iw++)
      {
        for (int id = 0; id < sample_per_part; id++)
        {
            float d = dstart + id * sub_bin_size_d;
            float w = wstart + iw * sub_bin_size_w;
            float h = hstart + ih * sub_bin_size_h;
            // bilinear interpolation
            if (w < -0.5 || w > width - 0.5 || h < -0.5 || h > height - 0.5 || d < -0.5 || d > depth - 0.5)
            {
            continue;
            }
            d = min(max(d, 0.), depth - 1.);
            w = min(max(w, 0.), width - 1.);
            h = min(max(h, 0.), height - 1.);
            int c = ((ctop * group_size + gh) * group_size + gw) * group_size + gd;
            float val = bilinear_interp(offset_bottom_data + c * height * width * depth, w, h, d, width, height, depth);
            sum += val;
            count++;
        }
      }
    }
    top_data[index] = count == 0 ? (float)(0) : sum / count;
    top_count[index] = count;
  }
}

__global__ void DeformablePSROIPoolBackwardAccKernel(
    const int count,
    const float *top_diff,
    const float *top_count,
    const int num_rois,
    const float spatial_scale,
    const int channels,
    const int height, const int width, const int depth,
    const int pooled_height, const int pooled_width, const int pooled_depth,
    const int output_dim,
    float *bottom_data_diff, float *bottom_trans_diff,
    const float *bottom_data,
    const float *bottom_rois,
    const float *bottom_trans,
    const int no_trans,
    const float trans_std,
    const int sample_per_part,
    const int group_size,
    const int part_size,
    const int num_classes,
    const int channels_each_class)
{
  CUDA_KERNEL_LOOP(index, count)
  {
    // The output is in order (n, ctop, ph, pw)
    int pd = index % pooled_depth;
    int pw = index % pooled_width;
    int ph = (index / pooled_depth / pooled_width) % pooled_height;
    int ctop = (index / pooled_depth / pooled_width / pooled_height) % output_dim;
    int n = index / pooled_depth / pooled_width / pooled_height / output_dim;

    // [start, end) interval for spatial sampling
    const float *offset_bottom_rois = bottom_rois + n * 7;
    int roi_batch_ind = offset_bottom_rois[0];
    float roi_start_d = (float)(round(offset_bottom_rois[1])) * spatial_scale - 0.5;
    float roi_start_w = (float)(round(offset_bottom_rois[2])) * spatial_scale - 0.5;
    float roi_start_h = (float)(round(offset_bottom_rois[3])) * spatial_scale - 0.5;
    float roi_end_d = (float)(round(offset_bottom_rois[4]) + 1.) * spatial_scale - 0.5;
    float roi_end_w = (float)(round(offset_bottom_rois[5]) + 1.) * spatial_scale - 0.5;
    float roi_end_h = (float)(round(offset_bottom_rois[6]) + 1.) * spatial_scale - 0.5;

    // Force too small ROIs to be 1x1
    float roi_depth = max(roi_end_d - roi_start_d, 0.1);
    float roi_width = max(roi_end_w - roi_start_w, 0.1); //avoid 0
    float roi_height = max(roi_end_h - roi_start_h, 0.1);

    // Compute w and h at bottom
    float bin_size_h = roi_height / (float)(pooled_height);
    float bin_size_w = roi_width / (float)(pooled_width);
    float bin_size_d = roi_depth / (float)(pooled_depth);

    float sub_bin_size_h = bin_size_h / (float)(sample_per_part);
    float sub_bin_size_w = bin_size_w / (float)(sample_per_part);
    float sub_bin_size_d = bin_size_d / (float)(sample_per_part);

    int part_h = floor((float)(ph) / pooled_height * part_size);
    int part_w = floor((float)(pw) / pooled_width * part_size);
    int part_d = floor((float)(pd) / pooled_depth * part_size);
    int class_id = ctop / channels_each_class;
    float trans_x = no_trans ? (float)(0) : bottom_trans[((((n * num_classes + class_id) * 3) * part_size + part_h) * part_size + part_w) * part_size + part_d] * trans_std;
    float trans_y = no_trans ? (float)(0) : bottom_trans[((((n * num_classes + class_id) * 3 + 1) * part_size + part_h) * part_size + part_w) * part_size + part_d] * trans_std;
    float trans_z = no_trans ? (float)(0) : bottom_trans[((((n * num_classes + class_id) * 3) * part_size + part_h) * part_size + part_w) * part_size + part_d] * trans_std;

    float dstart = (float)(pd)*bin_size_d + roi_start_d;
    dstart += trans_z * roi_depth;
    float wstart = (float)(pw)*bin_size_w + roi_start_w;
    wstart += trans_x * roi_width;
    float hstart = (float)(ph)*bin_size_h + roi_start_h;
    hstart += trans_y * roi_height;

    if (top_count[index] <= 0)
    {
      continue;
    }
    float diff_val = top_diff[index] / top_count[index];
    const float *offset_bottom_data = bottom_data + roi_batch_ind * channels * height * width * depth;
    float *offset_bottom_data_diff = bottom_data_diff + roi_batch_ind * channels * height * width * depth;
    int gd = floor((float)(pd)*group_size / pooled_depth);
    int gw = floor((float)(pw)*group_size / pooled_width);
    int gh = floor((float)(ph)*group_size / pooled_height);
    gd = min(max(gd, 0), group_size - 1);
    gw = min(max(gw, 0), group_size - 1);
    gh = min(max(gh, 0), group_size - 1);

    for (int ih = 0; ih < sample_per_part; ih++)
    {
      for (int iw = 0; iw < sample_per_part; iw++)
      {
        for (int id = 0; id < sample_per_part; id++)
        {
            float d = dstart + id * sub_bin_size_d;
            float w = wstart + iw * sub_bin_size_w;
            float h = hstart + ih * sub_bin_size_h;
            // bilinear interpolation
            if (w < -0.5 || w > width - 0.5 || h < -0.5 || h > height - 0.5 || d < -0.5 || d > depth - 0.5)
            {
            continue;
            }
            d = min(max(d, 0.), depth - 1.);
            w = min(max(w, 0.), width - 1.);
            h = min(max(h, 0.), height - 1.);
            int c = ((ctop * group_size + gh) * group_size + gw) * group_size + gd;
            // backward on feature
            int x0 = floor(w);
            int x1 = ceil(w);
            int y0 = floor(h);
            int y1 = ceil(h);
            int z0 = floor(d);
            int z1 = ceil(d);
            float dist_x = w - x0, dist_y = h - y0, dist_z = d - z0;
            float q000 = (1 - dist_x) * (1 - dist_y) * (1 - dist_z);
            float q010 = (1 - dist_x) * dist_y * (1 - dist_z);
            float q100 = dist_x * (1 - dist_y) * (1 - dist_z);
            float q110 = dist_x * dist_y * (1 - dist_z);
            float q001 = (1 - dist_x) * (1 - dist_y) * dist_z;
            float q011 = (1 - dist_x) * dist_y * dist_z;
            float q101 = dist_x * (1 - dist_y) * dist_z;
            float q111 = dist_x * dist_y * dist_z;
            int bottom_index_base = c * height * width * depth;
            atomicAdd(offset_bottom_data_diff + bottom_index_base + y0 * width + x0 + z0 * depth, q000 * diff_val);
            atomicAdd(offset_bottom_data_diff + bottom_index_base + y1 * width + x0 + z0 * depth, q010 * diff_val);
            atomicAdd(offset_bottom_data_diff + bottom_index_base + y0 * width + x1 + z0 * depth, q100 * diff_val);
            atomicAdd(offset_bottom_data_diff + bottom_index_base + y1 * width + x1 + z0 * depth, q110 * diff_val);
            atomicAdd(offset_bottom_data_diff + bottom_index_base + y0 * width + x0 + z1 * depth, q001 * diff_val);
            atomicAdd(offset_bottom_data_diff + bottom_index_base + y1 * width + x0 + z1 * depth, q011 * diff_val);
            atomicAdd(offset_bottom_data_diff + bottom_index_base + y0 * width + x1 + z1 * depth, q101 * diff_val);
            atomicAdd(offset_bottom_data_diff + bottom_index_base + y1 * width + x1 + z1 * depth, q111 * diff_val);

            if (no_trans)
            {
            continue;
            }
            float U000 = offset_bottom_data[bottom_index_base + y0 * width + x0 + z0 * depth];
            float U010 = offset_bottom_data[bottom_index_base + y1 * width + x0 + z0 * depth];
            float U100 = offset_bottom_data[bottom_index_base + y0 * width + x1 + z0 * depth];
            float U110 = offset_bottom_data[bottom_index_base + y1 * width + x1 + z0 * depth];
            float U001 = offset_bottom_data[bottom_index_base + y0 * width + x0 + z1 * depth];
            float U011 = offset_bottom_data[bottom_index_base + y1 * width + x0 + z1 * depth];
            float U101 = offset_bottom_data[bottom_index_base + y0 * width + x1 + z1 * depth];
            float U111 = offset_bottom_data[bottom_index_base + y1 * width + x1 + z1 * depth];
            float diff_x = ((U110 * (1 - dist_z) + U111 * dist_z) * dist_y + (U100 * (1 - dist_z) + U101 * dist_z) * (1 - dist_y) - (U010 * (1 - dist_z) + U011 * dist_z) * dist_y - (U000 * (1 - dist_z) + U001 * dist_z) * (1 - dist_y)) * trans_std * diff_val;
            diff_x *= roi_width;
            float diff_y = ((U110 * (1 - dist_z) + U111 * dist_z) * dist_x + (U010 * (1 - dist_z) + U011 * dist_z) * (1 - dist_x) - (U100 * (1 - dist_z) + U101 * dist_z) * dist_x - (U000 * (1 - dist_z) + U001 * dist_z) * (1 - dist_x)) * trans_std * diff_val;
            diff_y *= roi_height;
            float diff_z = ((U101 * (1 - dist_y) + U111 * dist_y) * dist_x + (U001 * (1 - dist_y) + U011 * dist_y) * (1 - dist_x) - (U100 * (1 - dist_y) + U110 * dist_y) * dist_x - (U000 * (1 - dist_y) + U010 * dist_y) * (1 - dist_x)) * trans_std * diff_val;
            diff_z *= roi_height;

            atomicAdd(bottom_trans_diff + ((((n * num_classes + class_id) * 3) * part_size + part_h) * part_size + part_w) * part_size + part_d, diff_z);
            atomicAdd(bottom_trans_diff + ((((n * num_classes + class_id) * 3) * part_size + part_h) * part_size + part_w) * part_size + part_d, diff_x);
            atomicAdd(bottom_trans_diff + ((((n * num_classes + class_id) * 3 + 1) * part_size + part_h) * part_size + part_w) * part_size + part_d, diff_y);
        }
      }
    }
  }
}

void DeformablePSROIPoolForward(cudaStream_t stream,
                                const float *data,
                                const float *bbox,
                                const float *trans,
                                float *out,
                                float *top_count,
                                const int batch,
                                const int channels,
                                const int height,
                                const int width,
                                const int depth,
                                const int num_bbox,
                                const int channels_trans,
                                const int no_trans,
                                const float spatial_scale,
                                const int output_dim,
                                const int group_size,
                                const int pooled_size,
                                const int part_size,
                                const int sample_per_part,
                                const float trans_std)
{

  const float *bottom_data = data;
  const float *bottom_rois = bbox;
  const float *bottom_trans = no_trans ? NULL : trans;
  float *top_data = out;
  float *top_count_data = top_count;

  const int pooled_height = pooled_size;
  const int pooled_width = pooled_size;
  const int pooled_depth = pooled_size;
  const int count = num_bbox * output_dim * pooled_height * pooled_width * pooled_depth;
  const int num_classes = no_trans ? 1 : channels_trans / 3;
  const int channels_each_class = no_trans ? output_dim : output_dim / num_classes;

  DeformablePSROIPoolForwardKernel<<<GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(
      count, bottom_data, spatial_scale, channels, height, width, depth, pooled_height, pooled_width, pooled_depth,
      bottom_rois, bottom_trans, no_trans, trans_std, sample_per_part, output_dim,
      group_size, part_size, num_classes, channels_each_class, top_data, top_count_data);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("error in DeformablePSROIPoolForward: %s\n", cudaGetErrorString(err));
  }
}

void DeformablePSROIPoolBackwardAcc(cudaStream_t stream,
                                    const float *out_grad,
                                    const float *data,
                                    const float *bbox,
                                    const float *trans,
                                    const float *top_count,
                                    float *in_grad,
                                    float *trans_grad,
                                    const int batch,
                                    const int channels,
                                    const int height,
                                    const int width,
                                    const int depth,
                                    const int num_bbox,
                                    const int channels_trans,
                                    const int no_trans,
                                    const float spatial_scale,
                                    const int output_dim,
                                    const int group_size,
                                    const int pooled_size,
                                    const int part_size,
                                    const int sample_per_part,
                                    const float trans_std)
{
  // LOG(INFO) << "DeformablePSROIPoolBackward";
  const float *top_diff = out_grad;
  const float *bottom_data = data;
  const float *bottom_rois = bbox;
  const float *bottom_trans = no_trans ? NULL : trans;
  float *bottom_data_diff = in_grad;
  float *bottom_trans_diff = no_trans ? NULL : trans_grad;
  const float *top_count_data = top_count;

  const int num_rois = num_bbox;
  const int pooled_height = pooled_size;
  const int pooled_width = pooled_size;
  const int pooled_depth = pooled_size;
  const int count = num_bbox * output_dim * pooled_height * pooled_width * pooled_depth;
  const int num_classes = no_trans ? 1 : channels_trans / 3;
  const int channels_each_class = no_trans ? output_dim : output_dim / num_classes;

  DeformablePSROIPoolBackwardAccKernel<<<GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(
      count, top_diff, top_count_data, num_rois, spatial_scale, channels, height, width, depth,
      pooled_height, pooled_width, pooled_depth, output_dim, bottom_data_diff, bottom_trans_diff,
      bottom_data, bottom_rois, bottom_trans, no_trans, trans_std, sample_per_part,
      group_size, part_size, num_classes, channels_each_class);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("error in DeformablePSROIPoolForward: %s\n", cudaGetErrorString(err));
  }
}