import torch
from torch.autograd import Function
from torch.nn.modules.utils import _pair, _triple

from _ext import deform_conv_2d, deform_conv_3d


def deform_conv_function(
    dim,
    input,
    offset,
    weight,
    stride=1,
    padding=0,
    dilation=1,
    deform_groups=1,
    im2col_step=64,
):

    if input.dim() != dim + 2:
        raise ValueError(
            "Expected {}D tensor as input, got {}D tensor instead.".format(
                dim + 2, input.dim()
            )
        )

    make_list = _triple if dim == 3 else _pair

    f = DeformConvFunction(
        make_list(stride),
        make_list(padding),
        make_list(dilation),
        deform_groups,
        im2col_step,
    )
    return f(input, offset, weight)


class DeformConvFunction(Function):
    def __init__(self, stride, padding, dilation, deformable_groups=1, im2col_step=64):
        super(DeformConvFunction, self).__init__()
        self.dim = len(stride)
        self.make_list = _triple if self.dim == 3 else _pair
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.deformable_groups = deformable_groups
        self.im2col_step = im2col_step

    def forward(self, input, offset, weight):
        self.save_for_backward(input, offset, weight)

        output = input.new(*self._output_size(input, weight))

        self.bufs_ = [*self.make_list(input.new())]  # columns, ones

        if not input.is_cuda:
            raise NotImplementedError
        else:
            if isinstance(input, torch.autograd.Variable):
                if not isinstance(input.data, torch.cuda.FloatTensor):
                    raise NotImplementedError
            else:
                if not isinstance(input, torch.cuda.FloatTensor):
                    raise NotImplementedError

            cur_im2col_step = min(self.im2col_step, input.shape[0])
            assert (
                input.shape[0] % cur_im2col_step
            ) == 0, "im2col step must divide batchsize"
            if self.dim == 3:
                deform_conv_2d.deform_conv_forward_cuda(
                    input,
                    weight,
                    offset,
                    output,
                    self.bufs_[0],
                    self.bufs_[1],
                    weight.size(3),
                    weight.size(2),
                    weight.size(4),
                    self.stride[1],
                    self.stride[0],
                    self.stride[2],
                    self.padding[1],
                    self.padding[0],
                    self.padding[2],
                    self.dilation[1],
                    self.dilation[0],
                    self.dilation[2],
                    self.deformable_groups,
                    cur_im2col_step,
                )
            else:
                deform_conv_2d.deform_conv_forward_cuda(
                    input,
                    weight,
                    offset,
                    output,
                    self.bufs_[0],
                    self.bufs_[1],
                    weight.size(3),
                    weight.size(2),
                    self.stride[1],
                    self.stride[0],
                    self.padding[1],
                    self.padding[0],
                    self.dilation[1],
                    self.dilation[0],
                    self.deformable_groups,
                    cur_im2col_step,
                )
        return output

    def backward(self, grad_output):
        input, offset, weight = self.saved_tensors

        grad_input = grad_offset = grad_weight = None

        if not grad_output.is_cuda:
            raise NotImplementedError
        else:
            if isinstance(grad_output, torch.autograd.Variable):
                if not isinstance(grad_output.data, torch.cuda.FloatTensor):
                    raise NotImplementedError
            else:
                if not isinstance(grad_output, torch.cuda.FloatTensor):
                    raise NotImplementedError

            cur_im2col_step = min(self.im2col_step, input.shape[0])
            assert (
                input.shape[0] % cur_im2col_step
            ) == 0, "im2col step must divide batchsize"

            if self.needs_input_grad[0] or self.needs_input_grad[1]:
                grad_input = input.new(*input.size()).zero_()
                grad_offset = offset.new(*offset.size()).zero_()
                if self.dim == 3:
                    deform_conv_3d.deform_conv_backward_input_cuda(
                        input,
                        offset,
                        grad_output,
                        grad_input,
                        grad_offset,
                        weight,
                        self.bufs_[0],
                        weight.size(3),
                        weight.size(2),
                        weight.size(4),
                        self.stride[1],
                        self.stride[0],
                        self.stride[2],
                        self.padding[1],
                        self.padding[0],
                        self.padding[2],
                        self.dilation[1],
                        self.dilation[0],
                        self.dilation[2],
                        self.deformable_groups,
                        cur_im2col_step,
                    )
                else:
                    deform_conv_2d.deform_conv_backward_input_cuda(
                        input,
                        offset,
                        grad_output,
                        grad_input,
                        grad_offset,
                        weight,
                        self.bufs_[0],
                        weight.size(3),
                        weight.size(2),
                        self.stride[1],
                        self.stride[0],
                        self.padding[1],
                        self.padding[0],
                        self.dilation[1],
                        self.dilation[0],
                        self.deformable_groups,
                        cur_im2col_step,
                    )

            if self.needs_input_grad[2]:
                grad_weight = weight.new(*weight.size()).zero_()
                if self.dim == 3:
                    deform_conv_3d.deform_conv_backward_parameters_cuda(
                        input,
                        offset,
                        grad_output,
                        grad_weight,
                        self.bufs_[0],
                        self.bufs_[1],
                        weight.size(3),
                        weight.size(2),
                        weight.size(3),
                        self.stride[1],
                        self.stride[0],
                        self.stride[2],
                        self.padding[1],
                        self.padding[0],
                        self.padding[2],
                        self.dilation[1],
                        self.dilation[0],
                        self.dilation[2],
                        self.deformable_groups,
                        1,
                        cur_im2col_step,
                    )
                else:
                    deform_conv_2d.deform_conv_backward_parameters_cuda(
                        input,
                        offset,
                        grad_output,
                        grad_weight,
                        self.bufs_[0],
                        self.bufs_[1],
                        weight.size(3),
                        weight.size(2),
                        self.stride[1],
                        self.stride[0],
                        self.padding[1],
                        self.padding[0],
                        self.dilation[1],
                        self.dilation[0],
                        self.deformable_groups,
                        1,
                        cur_im2col_step,
                    )

        return grad_input, grad_offset, grad_weight

    def _output_size(self, input, weight):
        channels = weight.size(0)

        output_size = (input.size(0), channels)
        for d in range(input.dim() - 2):
            in_size = input.size(d + 2)
            pad = self.padding[d]
            kernel = self.dilation[d] * (weight.size(d + 2) - 1) + 1
            stride = self.stride[d]
            output_size += ((in_size + (2 * pad) - kernel) // stride + 1,)
        if not all(map(lambda s: s > 0, output_size)):
            raise ValueError(
                "convolution input is too small (output would be {})".format(
                    "x".join(map(str, output_size))
                )
            )
        return output_size
