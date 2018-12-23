import torch
from torch.autograd import Function
from _ext import roi_pooling
import pdb

class RoIOffsetPoolFunction(Function):
    def __init__(ctx, pooled_height, pooled_width, spatial_scale):
        ctx.pooled_width = pooled_width
        ctx.pooled_height = pooled_height
        ctx.spatial_scale = spatial_scale
        ctx.feature_size = None

    def forward(ctx, features, rois, offset): 
        ctx.feature_size = features.size()
        ctx.features = features
        batch_size, num_channels, data_height, data_width = ctx.feature_size
        ctx.num_rois = rois.size(0)
        output = features.new(ctx.num_rois, num_channels, ctx.pooled_height, ctx.pooled_width).zero_()
        ctx.argmax = features.new(ctx.num_rois, num_channels, ctx.pooled_height, ctx.pooled_width).zero_().int()
        ctx.rois = rois
        
        offset = offset.unsqueeze(1).repeat(1, num_channels, 1, 1, 1)
        ctx.offset = offset
        if not features.is_cuda:
            _features = features.permute(0, 2, 3, 1)
            roi_pooling.roioffset_pooling_forward(ctx.pooled_height, ctx.pooled_width, ctx.spatial_scale,
                                            _features, rois, offset, output)
        else:
            roi_pooling.roioffset_pooling_forward_cuda(ctx.pooled_height, ctx.pooled_width, ctx.spatial_scale,
                                                 features, rois, offset, output, ctx.argmax)
        
        return output

    def backward(ctx, grad_output):
        assert(ctx.feature_size is not None and grad_output.is_cuda)
        
        batch_size, num_channels, data_height, data_width = ctx.feature_size
        grad_input = grad_output.new(batch_size, num_channels, data_height, data_width).zero_()
        grad_offset = grad_output.new(ctx.num_rois, num_channels, ctx.pooled_height, ctx.pooled_width, 2).zero_()

        roi_pooling.roioffset_pooling_backward_cuda(ctx.pooled_height, ctx.pooled_width, ctx.spatial_scale, grad_output, ctx.rois, ctx.offset, grad_input, grad_offset, ctx.argmax, ctx.features)

        print("grad offset shape: {}".format(grad_offset.shape))
        grad_offset = grad_offset.sum(1) # sum on the channel index

        return grad_input, None, grad_offset