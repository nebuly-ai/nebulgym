from typing import Any, Optional

import torch
import torch.nn.grad
import torch.nn.functional as F
from torch.autograd.graph import saved_tensors_hooks
from torch.nn.modules.utils import _pair


class ReLUFunction(torch.autograd.Function):
    @staticmethod
    def jvp(ctx: Any, *grad_inputs: Any) -> Any:
        pass

    @staticmethod
    def forward(ctx, x: torch.Tensor):
        with torch.no_grad():
            y = F.relu(x)
            binary = y.bool()
            ctx.save_for_backward(binary)
            return y

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        with torch.no_grad():
            binary = ctx.saved_tensors[0]
            return dy * binary


def _compute_quantization_stats(input):
    scale = input.abs().max() / 127
    q = -(input.mean() // 127)
    return scale, q


relu_func = ReLUFunction.apply


class GistReLU(torch.nn.ReLU):
    def forward(self, x: torch.Tensor):
        return relu_func(x)


def _quantize_input(input):
    scale, q = _compute_quantization_stats(input)
    input = torch.quantize_per_tensor(input, scale, q, dtype=torch.qint8)
    return input


def _dequantize_input(input):
    return input.dequantize()


class InputQuantizationPackingState:
    def __init__(self):
        self._state = 0

    def pack_hook(self, packed_tensor):
        if self._state == 0:
            self._state += 1
            return _quantize_input(packed_tensor)
        return packed_tensor

    def unpack_hook(self, packed_tensor):
        if self._state > 0:
            self._state = 0
            return _dequantize_input(packed_tensor)
        return packed_tensor


class GistQuantizedConv2d(torch.nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_quantization_packing = InputQuantizationPackingState()

    def _conv_forward(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
    ):
        if self.padding_mode != "zeros":
            return F.conv2d(
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
        return F.conv2d(
            input,
            weight,
            bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        with saved_tensors_hooks(
            self.input_quantization_packing.pack_hook,
            self.input_quantization_packing.unpack_hook,
        ):
            return self._conv_forward(input, self.weight, self.bias)


class GistQuantizedLinear(torch.nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_quantization_packing = InputQuantizationPackingState()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        with saved_tensors_hooks(
            self.input_quantization_packing.pack_hook,
            self.input_quantization_packing.unpack_hook,
        ):
            return F.linear(input, self.weight, self.bias)


class GistQuantizedBatchNorm2d(torch.nn.BatchNorm2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_quantization_packing = InputQuantizationPackingState()

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
        with saved_tensors_hooks(
            self.input_quantization_packing.pack_hook,
            self.input_quantization_packing.unpack_hook,
        ):
            return F.batch_norm(
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
    if False and isinstance(module, torch.nn.Linear):
        new_module = GistQuantizedLinear(
            module.in_features,
            module.out_features,
            module.bias is not None,
            device=module.weight.device,
        )
        new_module.weight = module.weight
        new_module.bias = module.bias
    elif False and isinstance(module, torch.nn.Conv2d):
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
    elif isinstance(module, torch.nn.BatchNorm2d):
        new_module = GistQuantizedBatchNorm2d(
            module.num_features,
            module.eps,
            module.momentum,
            module.affine,
            module.track_running_stats,
        )
        new_module.weight = module.weight
        new_module.bias = module.bias
    else:
        named_children = list(module.named_children())
        if len(named_children) > 0:
            for name, submodule in named_children:
                new_submodule = gist_modify_network(submodule)
                if new_submodule is not module:
                    module._modules[name] = new_submodule

        new_module = module
    return new_module
