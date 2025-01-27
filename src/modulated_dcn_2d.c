#include <TH/TH.h>
#include <stdio.h>
#include <math.h>

void modulated_deform_conv_forward(THFloatTensor *input, THFloatTensor *weight,
                                   THFloatTensor *bias, THFloatTensor *ones,
                                   THFloatTensor *offset, THFloatTensor *mask,
                                   THFloatTensor *output, THFloatTensor *columns,
                                   const int pad_h, const int pad_w,
                                   const int stride_h, const int stride_w,
                                   const int dilation_h, const int dilation_w,
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
                                    int kernel_h, int kernel_w,
                                    int stride_h, int stride_w,
                                    int pad_h, int pad_w,
                                    int dilation_h, int dilation_w,
                                    int deformable_group)
{
    printf("only implemented in GPU");
}