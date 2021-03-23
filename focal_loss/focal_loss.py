import torch
import focal_loss_cuda


class FocalLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, cls_output, cls_targets_at_level, num_positives_sum, num_classes,
                alpha, gamma, label_smoothing=0.0):
        if isinstance(num_positives_sum, torch.Tensor):
            num_positives_sum = num_positives_sum.item()
        num_positives_sum = int(num_positives_sum)
        loss, partial_grad = focal_loss_cuda.forward(cls_output,
                                                     cls_targets_at_level,
                                                     num_positives_sum,
                                                     num_classes, alpha, gamma,
                                                     label_smoothing)

        ctx.save_for_backward(partial_grad)
        return loss

    @staticmethod
    def backward(ctx, grad_loss):
        partial_grad, = ctx.saved_tensors

        # The backward kernel is actually in-place to save memory space, partial_grad
        # and grad_input are the same tensor.
        grad_input = focal_loss_cuda.backward(grad_loss, partial_grad)

        return grad_input, None, None, None, None, None, None
