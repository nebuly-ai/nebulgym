from typing import Any, Tuple

import torch
import torch.nn.functional as F


# TODO: MODIFY THE BACKPROP WITH APPROXIMATION TECHNIQUES
class NebulBatchNormFunc(torch.autograd.Function):
    @staticmethod
    def jvp(ctx: Any, *grad_inputs: Any) -> Any:
        pass

    @staticmethod
    def forward(
        ctx,
        input,
        running_mean,
        running_var,
        weight,
        bias,
        bn_training,
        exponential_average_factor,
        eps,
    ):
        with torch.no_grad():
            ctx.save_for_backward(
                input,
                weight,
                bias,
                eps,
            )
            return F.batch_norm(
                input,
                running_mean,
                running_var,
                weight,
                bias,
                bn_training,
                exponential_average_factor,
                eps,
            )

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[Any]:
        with torch.no_grad():
            input, weight, bias, eps = ctx.saved_tensors
            input = input.to_dense()
            input_grad = weight_grad = bias_grad = None
            mu = torch.mean(input, dim=(0, 2, 3), keepdim=True)
            inv_sqrt = 1 / torch.sqrt(
                torch.mean(
                    torch.square(input) - torch.square(mu),
                    dim=(0, 2, 3),
                    keepdim=True,
                )
                + eps
            )
            x_hat = (input - mu) * inv_sqrt
            if ctx.needs_input_grad[0]:
                shape = input.shape
                M = shape[0] * shape[2] * shape[3]
                dx_hat_dx = (1 - 1 / M) * inv_sqrt - (
                    input - mu / M
                ) * x_hat * torch.square(inv_sqrt)
                input_grad = weight * grad_output * dx_hat_dx
            if ctx.needs_input_grad[3]:
                weight_grad = torch.sum(grad_output * x_hat, dim=(0, 2, 3))
            if ctx.needs_input_grad[4]:
                bias_grad = torch.sum(grad_output, dim=(0, 2, 3))
            return (
                input_grad,
                None,
                None,
                weight_grad,
                bias_grad,
                None,
                None,
                None,
            )


nebul_batch_norm_func = NebulBatchNormFunc.apply


class NebulBatchNorm2d(torch.nn.BatchNorm2d):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        self._check_input_dim(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip
            #  emitting this when it is None
            if self.num_batches_tracked is not None:  # type: ignore[has-type]
                self.num_batches_tracked.add_(1)  # type: ignore[has-type]
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(
                        self.num_batches_tracked
                    )
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (
                self.running_var is None
            )

        return nebul_batch_norm_func(
            input,
            # If buffers are not to be tracked, ensure that they won't
            # be updated
            self.running_mean
            if not self.training or self.track_running_stats
            else None,
            self.running_var
            if not self.training or self.track_running_stats
            else None,
            self.weight,
            self.bias,
            bn_training,
            exponential_average_factor,
            self.eps,
        )
