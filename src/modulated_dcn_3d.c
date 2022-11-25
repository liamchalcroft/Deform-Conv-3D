#include <TH/TH.h>
#include <stdio.h>
#include <math.h>

void modulated_deform_conv_forward(THFloatTensor *input, THFloatTensor *weight,
                                   THFloatTensor *bias, THFloatTensor *ones,
                                   THFloatTensor *offset, THFloatTensor *mask,
                                   THFloatTensor *output, THFloatTensor *columns,
                                   const int pad_h, const int pad_w, const int pad_d,
                                   const int stride_h, const int stride_w, const int stride_d,
                                   const int dilation_h, const int dilation_w, const int dilation_d,
                                   const int deformable_group)
{
    printf("only implemented in GPU");
}
void modulated_deform_conv_backward(THFloatTensor *input, THFloatTensor *weight,
                                    THFloatTensor *bias, THFloatTensor *ones,
                                    THFloatTensor *offset, THFloatTensor *mask,
                                    THFloatTensor *output, THFloatTensor *columns,
                                    THFloatTensor *grad_input, THFloatTensor *grad_weight,
                                    THFloatTensor *grad_bias, THFloatTensor *grad_offset,
                                    THFloatTensor *grad_mask, THFloatTensor *grad_output,
                                    int kernel_h, int kernel_w, int kernel_d,
                                    int stride_h, int stride_w, int stride_d,
                                    int pad_h, int pad_w, int pad_d,
                                    int dilation_h, int dilation_w, int dilation_d,
                                    int deformable_group)
{
    printf("only implemented in GPU");
}