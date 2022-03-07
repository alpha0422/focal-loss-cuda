import torch
import focal_loss_cuda


class FocalLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, cls_output, cls_targets_at_level, num_positives_sum,
                num_real_classes, alpha, gamma, label_smoothing=0.0):
        loss, partial_grad = focal_loss_cuda.forward(cls_output,
                                                     cls_targets_at_level,
                                                     num_positives_sum,
                                                     num_real_classes,
                                                     alpha, gamma,
                                                     label_smoothing)

        ctx.save_for_backward(partial_grad, num_positives_sum)
        return loss

    @staticmethod
    def backward(ctx, grad_loss):
        partial_grad, num_positives_sum = ctx.saved_tensors

        # The backward kernel is actually in-place to save memory space,
        # partial_grad and grad_input are the same tensor.
        grad_input = focal_loss_cuda.backward(grad_loss, partial_grad,
                                              num_positives_sum)

        return grad_input, None, None, None, None, None, None


class FocalBCELoss(torch.autograd.Function):
    """
    FocalBCELoss expects y_true as one-hot tensor, and no final reduction.
    """

    @staticmethod
    def forward(ctx, y_pred, y_true, alpha, gamma):
        loss, partial_grad = focal_loss_cuda.bce_forward(y_pred, y_true,
                                                         alpha, gamma)
        ctx.save_for_backward(partial_grad)
        return loss

    @staticmethod
    def backward(ctx, grad_loss):
        partial_grad, = ctx.saved_tensors

        # Loss is often followed by sum(), thus stride of grad_loss is (0, 0).
        # TODO: remove this once the reduction is included in the CUDA
        # implementation.
        grad_loss = grad_loss.contiguous()

        # The backward kernel is actually in-place to save memory space,
        # partial_grad and grad_input are the same tensor.
        grad_input = focal_loss_cuda.bce_backward(grad_loss, partial_grad)

        return grad_input, None, None, None
