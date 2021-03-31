import torch
import focal_loss_cuda


class FocalLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, cls_output, cls_targets_at_level, num_positives_sum,
                num_real_classes, alpha, gamma, label_smoothing=0.0):
        loss = focal_loss_cuda.forward(cls_output,
                                       cls_targets_at_level,
                                       num_positives_sum,
                                       num_real_classes,
                                       alpha, gamma,
                                       label_smoothing)

        ctx.save_for_backward(cls_output, cls_targets_at_level,
                              num_positives_sum,
                              torch.IntTensor([num_real_classes]),
                              torch.FloatTensor([alpha]),
                              torch.FloatTensor([gamma]),
                              torch.FloatTensor([label_smoothing]))

        return loss

    @staticmethod
    def backward(ctx, grad_loss):
        cls_output, cls_targets_at_level, num_positives_sum, num_real_classes, \
            alpha, gamma, label_smoothing = ctx.saved_tensors
        num_real_classes = num_real_classes.item()
        alpha = alpha.item()
        gamma = gamma.item()
        label_smoothing = label_smoothing.item()

        grad_input = focal_loss_cuda.backward(grad_loss,
                                              cls_output,
                                              cls_targets_at_level,
                                              num_positives_sum,
                                              num_real_classes,
                                              alpha, gamma,
                                              label_smoothing)

        return grad_input, None, None, None, None, None, None
