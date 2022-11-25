int deform_conv_forward_cuda(THCudaTensor *input,
                             THCudaTensor *weight, /*THCudaTensor * bias, */
                             THCudaTensor *offset, THCudaTensor *output,
                             THCudaTensor *columns, THCudaTensor *ones, int kW,
                             int kH, int kD, int dW, int dH, int dD, int padW, int padH, int padD,
                             int dilationW, int dilationH, int dilationD,
                             int deformable_group, int im2col_step);

int deform_conv_backward_input_cuda(
    THCudaTensor *input, THCudaTensor *offset, THCudaTensor *gradOutput,
    THCudaTensor *gradInput, THCudaTensor *gradOffset, THCudaTensor *weight,
    THCudaTensor *columns, int kW, int kH, int kD, int dW, int dH, int dD, int padW, int padH, int padD,
    int dilationW, int dilationH, int dilationD, int deformable_group, int im2col_step);

int deform_conv_backward_parameters_cuda(
    THCudaTensor *input, THCudaTensor *offset, THCudaTensor *gradOutput,
    THCudaTensor *gradWeight, /*THCudaTensor *gradBias, */
    THCudaTensor *columns, THCudaTensor *ones, int kW, int kH, int kD, int dW, int dH, int dD,
    int padW, int padH, int padD, int dilationW, int dilationH, int dilationD, int deformable_group,
    float scale, int im2col_step);
