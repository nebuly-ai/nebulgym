from typing import Any, Optional, Tuple

import torch
import torch.nn.grad
import torch.nn.functional as F
from torch.nn.modules.utils import _pair


class ReLUFunction(torch.autograd.Function):
    @staticmethod
    def jvp(ctx: Any, *grad_inputs: Any) -> Any:
        pass

    @staticmethod
    def forward(ctx, x: torch.Tensor):
        y = F.relu(x)
        binary = y.bool()
        ctx.save_for_backward(binary)
        return y

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        binary = ctx.saved_tensors[0]
        return dy * binary


class PostReLUConv2dFunction(torch.autograd.Function):
    @staticmethod
    def jvp(ctx: Any, *grad_inputs: Any) -> Any:
        pass

    @staticmethod
    def forward(cxt, input, weight, bias, stride, padding, dilation, groups):
        cxt.save_for_backward(
            input.view(-1, input.shape[-1]).to_sparse_csr(), weight, bias
        )
        # cxt.save_for_backward(input.to_sparse_coo(), weight, bias)
        return F.conv2d(input, weight, bias, stride, padding, dilation, groups)

    @staticmethod
    def backward(cxt, grad_output):
        input, weight, bias = cxt.saved_tensors
        input = input.to_dense().view(
            grad_output.shape[0],
            weight.shape[1],
            -1,
            input.shape[1],
        )
        # input = input.to_dense()
        grad_input = grad_weight = grad_bias = None

        if cxt.needs_input_grad[0]:
            grad_input = torch.nn.grad.conv2d_input(
                input.shape, weight, grad_output
            )

        if cxt.needs_input_grad[1]:
            grad_weight = torch.nn.grad.conv2d_weight(
                input, weight.shape, grad_output
            )

        if bias is not None and cxt.needs_input_grad[2]:
            grad_bias = grad_output.sum((0, 2, 3)).squeeze(0)

        if bias is not None:
            return grad_input, grad_weight, grad_bias, None, None, None, None
        else:
            return grad_input, grad_weight, None, None, None, None, None


def _compute_quantization_stats(input):
    scale = input.abs().max() / 127
    q = -(input.mean() // 127)
    return scale, q


class QuantizedConv2dFunction(torch.autograd.Function):
    @staticmethod
    def jvp(ctx: Any, *grad_inputs: Any) -> Any:
        pass

    @staticmethod
    def forward(
        cxt: Any,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        stride: int,
        padding: int,
        dilation: int,
        groups: int,
    ):
        scale, q = _compute_quantization_stats(input)
        cxt.save_for_backward(
            torch.quantize_per_tensor(input, scale, q, dtype=torch.qint8),
            weight,
            bias,
        )
        return F.conv2d(input, weight, bias, stride, padding, dilation, groups)

    @staticmethod
    def backward(cxt, grad_output):
        input, weight, bias = cxt.saved_tensors

        input = torch.dequantize(input)
        grad_input = grad_weight = grad_bias = None

        if cxt.needs_input_grad[0]:
            grad_input = torch.nn.grad.conv2d_input(
                input.shape, weight, grad_output
            )

        if cxt.needs_input_grad[1]:
            # grad_weight = torch.nn.grad.conv2d_weight(
            #     input, weight.shape, grad_output
            # )
            # grad_weight = conv2d_weight(input, weight.shape, grad_output)
            grad_weight = F.conv2d(
                input.transpose(0, 1), grad_output.transpose(0, 1)
            ).transpose(0, 1)

        if bias is not None and cxt.needs_input_grad[2]:
            grad_bias = grad_output.sum((0, 2, 3)).squeeze(0)

        if bias is not None:
            return grad_input, grad_weight, grad_bias, None, None, None, None
        else:
            return grad_input, grad_weight, None, None, None, None, None


class QuantizedLinearFunction(torch.autograd.Function):
    @staticmethod
    def jvp(ctx: Any, *grad_inputs: Any) -> Any:
        pass

    @staticmethod
    def forward(
        ctx: Any,
        input_tensor: torch.Tensor,
        weights: torch.Tensor,
        bias: Optional[torch.Tensor],
    ) -> torch.Tensor:
        scale, q = _compute_quantization_stats(input_tensor)
        ctx.save_for_backward(
            torch.quantize_per_tensor(input_tensor, scale, q, torch.qint8),
            weights,
            bias,
        )
        return F.linear(input_tensor, weights, bias)

    @staticmethod
    def backward(
        ctx: Any, grad_output: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        q_input, weight, bias = ctx.saved_tensors
        input = torch.dequantize(q_input)
        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = torch.matmul(grad_output, weight)
        if ctx.needs_input_grad[1]:
            grad_weight = torch.matmul(grad_output.t(), input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.reshape(-1, grad_output.shape[-1]).sum(0)

        return grad_input, grad_weight, grad_bias


class QuantizedGistBatchNormFunc(torch.autograd.Function):
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
            scale, q = _compute_quantization_stats(input)
            ctx.save_for_backward(
                torch.quantize_per_tensor(input, scale, q, dtype=torch.qint8),
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
        return input_grad, None, None, weight_grad, bias_grad, None, None, None


relu_func = ReLUFunction.apply
post_relu_conv_func = PostReLUConv2dFunction.apply
quantized_conv_func = QuantizedConv2dFunction.apply
quantized_linear_func = QuantizedLinearFunction.apply
quantized_batchnorm_func = QuantizedGistBatchNormFunc.apply


class GistReLU(torch.nn.ReLU):
    def forward(self, x: torch.Tensor):
        return relu_func(x)


class GistPostReLUConv2d(torch.nn.Conv2d):
    def _conv_forward(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
    ):
        if self.padding_mode != "zeros":
            return post_relu_conv_func(
                F.pad(
                    input,
                    self._reversed_padding_repeated_twice,
                    mode=self.padding_mode,
                ),
                weight,
                bias,
                self.stride,
                _pair(0),
                self.dilation,
                self.groups,
            )
        return post_relu_conv_func(
            input,
            weight,
            bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self._conv_forward(input, self.weight, self.bias)


class GistQuantizedConv2d(torch.nn.Conv2d):
    def _conv_forward(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
    ):
        if self.padding_mode != "zeros":
            return quantized_conv_func(
                F.pad(
                    input,
                    self._reversed_padding_repeated_twice,
                    mode=self.padding_mode,
                ),
                weight,
                bias,
                self.stride,
                _pair(0),
                self.dilation,
                self.groups,
            )
        return quantized_conv_func(
            input,
            weight,
            bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self._conv_forward(input, self.weight, self.bias)


class GistQuantizedLinear(torch.nn.Linear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return quantized_linear_func(input, self.weight, self.bias)


class GistQuantizedBatchNorm2d(torch.nn.BatchNorm2d):
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

        return quantized_batchnorm_func(
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


def gist_modify_network(module: torch.nn.Module):
    if isinstance(module, torch.nn.Linear):
        new_module = GistQuantizedLinear(
            module.in_features,
            module.out_features,
            module.bias is not None,
            device=module.weight.device,
        )
        new_module.weight = module.weight
        new_module.bias = module.bias
    elif isinstance(module, torch.nn.Conv2d):
        new_module = GistQuantizedConv2d(
            module.in_channels,
            module.out_channels,
            module.kernel_size,
            module.stride,
            module.padding,
            module.dilation,
            module.groups,
            module.bias is not None,
            device=module.weight.device,
        )
        new_module.weight = module.weight
        new_module.bias = new_module.bias
    elif isinstance(module, torch.nn.ReLU):
        new_module = GistReLU(inplace=module.inplace)
    else:
        named_children = list(module.named_children())
        if len(named_children) > 0:
            for name, submodule in named_children:
                new_submodule = gist_modify_network(submodule)
                if new_submodule is not module:
                    module._modules[name] = new_submodule

        new_module = module
    return new_module
