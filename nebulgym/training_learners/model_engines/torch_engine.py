from typing import Tuple, Union

import torch.nn

from nebulgym.training_learners.model_engines.base import BaseEngine
from nebulgym.training_learners.model_engines.utils import (
    CallMethodFixingEnvRunner,
)


class TorchEngine(BaseEngine):
    """DL Engine running on top of pytorch. It is a wrapper around the pytorch
    model aiming at providing the same Interface for all the engines.

    Attributes:
        model (Module): The pytorch model
    """

    def __init__(self, model: torch.nn.Module):
        super().__init__(model)

    def run(self, *args, **kwargs):
        return self._model.forward(*args, **kwargs)

    def eval(self, update_torch_model: bool = True):
        if update_torch_model:
            self._model.eval()
        return self

    def train(self, update_torch_model: bool = True):
        if update_torch_model:
            self._model.train()
        return self


class TorchScriptEngine(BaseEngine):
    def __init__(self, model: torch.nn.Module):
        super().__init__(model)
        self._optimized_model = None

    @staticmethod
    def _get_compiled_model(
        model: torch.nn.Module,
        inputs: Union[Tuple[torch.Tensor, ...], torch.Tensor],
    ):
        with CallMethodFixingEnvRunner(model):
            traced_module = torch.jit.trace(model, example_inputs=inputs)
        return traced_module

    def run(self, *args, **kwargs):
        if self._optimized_model is None:
            if len(kwargs) > 0 or any(
                not isinstance(arg, torch.Tensor) for arg in args
            ):
                raise ValueError(
                    "Keywords and non-tensor arguments are not taken in "
                    "consideration while tracing the model and should not be "
                    "given. For avoiding errors in the following this error "
                    "is generated."
                )
            self._optimized_model = self._get_compiled_model(self._model, args)
        return self._optimized_model.forward(*args, **kwargs)

    def eval(self, update_torch_model: bool = True):
        if self._optimized_model is not None:
            self._optimized_model.eval()
        elif update_torch_model:
            self._model.eval()
        return self

    def train(self, update_torch_model: bool = True):
        if self._optimized_model is not None:
            self._optimized_model.train()
        elif update_torch_model:
            self._model.train()
        return self
