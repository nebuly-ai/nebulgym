import warnings
from logging import Logger
from typing import List, Optional, Tuple, Generator

import torch.nn
from torch.nn import Module

from nebulgym.base import NebulgymBackend
from nebulgym.training_learners.model_engines import ENGINE_MAP, BaseEngine
from nebulgym.training_learners.model_engines.torch_engine import TorchEngine


def _try_to_instantiate(
    model: torch.nn.Module,
    backends: List[NebulgymBackend],
    logger: Logger = None,
) -> Generator[Tuple[NebulgymBackend, Optional[BaseEngine], bool], None, None]:
    for backend in backends:
        try:
            engine = ENGINE_MAP[backend](model)
            yield backend, engine, True
        except Exception as ex:
            error_msg = (
                f"Got error {ex} while instantiating engine for {backend}. "
                f"The backend will be ignored."
            )
            if logger is not None:
                logger.warning(error_msg)
            else:
                warnings.warn(error_msg)
            yield backend, None, False


class TrainingLearner:
    """Core class used as model interface for all Nebulgym optimization
    techniques. The choice and management of all the different backends that
    can be used at RunTime is delegated to this class.

    Attributes:
        model (Module): The original pytorch model
        backends: The list of backends that can be used for running the model.
            Available backends are "ONNXRUNTIME", "RAMMER" and "PYTORCH".
        logger (Logger, optional): Logger to be used for printing the generated
            logs.
        extend (bool, optional): Boolean flag to be used when the
            TrainingLearner is instantiated in the decorators framework. It
            prevents the annotated pytorch class to call recursively its
            methods. It needs to be put to False when used inside a
            class decorator.
    """

    def __init__(
        self,
        model: Module,
        backends: List[NebulgymBackend] = None,
        logger: Logger = None,
        extend: bool = True,
    ):
        backends = (
            [NebulgymBackend(b) for b in backends]
            if backends
            else [NebulgymBackend.TORCHSCRIPT]
        )
        self._backend_engines = (
            {
                b: engine
                for b, engine, ok in _try_to_instantiate(
                    model, backends, logger
                )
                if ok
            }
            if len(backends) > 0
            else {}
        )
        self._backend_priorities = backends
        self._default_backend = TorchEngine(model)
        self._running_backend: Optional[BaseEngine] = None
        self._is_training = False
        self._logger = logger
        self._extend = extend

    def train(self):
        self._is_training = True
        for _, engine in self._backend_engines.items():
            engine.train(update_torch_model=self._extend)
        self._default_backend.train(update_torch_model=self._extend)
        self._running_backend = None
        return self

    def eval(self):
        self._is_training = False
        for _, engine in self._backend_engines.items():
            engine.eval(update_torch_model=self._extend)
        self._default_backend.eval(update_torch_model=self._extend)

    def __call__(self, *args, **kwargs):
        if self._is_training and self._running_backend is not None:
            return self._running_backend.run(*args, **kwargs)
        for b in self._backend_priorities:
            try:
                res = self._backend_engines[b].run(*args, **kwargs)
                if self._is_training:
                    self._running_backend = self._backend_engines[b]
                return res
            except Exception as ex:
                error_msg = (
                    f"Error with backend {b}. Got error {ex}. "
                    f"Switching to the next one."
                )
                if self._logger is not None:
                    self._logger.warning(error_msg)
                else:
                    warnings.warn(error_msg)
        if self._is_training:
            self._running_backend = self._default_backend
        return self._default_backend.run(*args, **kwargs)

    def __getattr__(self, item):
        # All the backend objects point to the original pytorch model saved
        # into the _default_backend attribute.
        if not self.__dict__.get("_extend", False):
            raise AttributeError(
                "For avoiding infinite recursion the attributes of the inner "
                "model are not directly accessible from the TrainingLearner. "
                "You can still access the inner model trough the attribute "
                "containing the default backend"
            )
        return getattr(self._default_backend.original_model, item)
