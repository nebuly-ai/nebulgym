from typing import Type

import torch.nn
from torch.utils.data import Dataset

from nebulgym.data.nebuly_dataset import NebulDataset
from nebulgym.patches.pytorch.meprop_linear import patch_backward_pass
from nebulgym.training_learners.base import TrainingLearner


def _new_module_call(self: torch.nn.Module, *args, **kwargs):
    if getattr(self, "_nebulgym_backend", None) is not None:
        return self._nebulgym_backend(*args, **kwargs)
    return torch.nn.Module.__call__(self, *args, **kwargs)


def _patch_module_call(cls: Type[torch.nn.Module]):
    cls.__call__ = _new_module_call
    return cls


def _patch_module_train_and_eval(cls: Type[torch.nn.Module]):
    previous_train = cls.train
    previous_eval = cls.eval

    def _new_eval(self, *args, **kwargs):
        self = previous_eval(self, *args, **kwargs)
        nebulgym_backend = getattr(self, "_nebulgym_backend", None)
        if nebulgym_backend is not None:
            nebulgym_backend.eval()
        return self

    def _new_train(self, *args, **kwargs):
        self = previous_train(self, *args, **kwargs)
        nebulgym_backend = getattr(self, "_nebulgym_backend", None)
        if nebulgym_backend is not None:
            nebulgym_backend.train()
        return self

    cls.train = _new_train
    cls.eval = _new_eval
    return cls


def _patch_module_init(
    cls: Type[torch.nn.Module], patch_backprop: bool, **nebulgym_kwargs
):
    prev_init = cls.__init__

    def _new_init(self: torch.nn.Module, *args, **kwargs):
        prev_init(self, *args, **kwargs)
        if patch_backprop:
            self = patch_backward_pass(self)
        setattr(
            self,
            "_nebulgym_backend",
            TrainingLearner(self, extend=False, **nebulgym_kwargs),
        )

    cls.__init__ = _new_init
    return cls


def accelerate_model(patch_backprop: bool = True, **nebulgym_kwargs):
    """Function that may be used as class decorator. The decorator patches
    the input class, modifying the __init__, __call__, train and eval methods.
    The modified version switches the backend of the calculations to the
    TrainingLearner defined in nebulgym.

    Args:
        patch_backprop (bool, optional): Boolean flag for adopting the backprop
            optimization techniques. This kind of techniques can slightly
            affect the model final performance. However, they usually lead to
            a similar effect to dropout: the network regularization, i.e.
            better performance.
        **nebulgym_kwargs (Dict): Extra parameters that must be passed to the
            TrainingLearner. See the TrainingLearner description for further
            information.

    Returns:
        the class decorator to be applied to the torch.nn.Module.
    """

    def _inner_patch(cls: Type[torch.nn.Module]):
        if cls.__call__ is _new_module_call:  # class already patched
            return cls
        cls = _patch_module_init(cls, patch_backprop, **nebulgym_kwargs)
        cls = _patch_module_call(cls)
        cls = _patch_module_train_and_eval(cls)
        return cls

    return _inner_patch


def accelerate_dataset(**nebuldata_kwargs):
    """Function that may be used as class decorator. The class decorator
    patches a pytorch's Dataset class defined by the user. It modifies the
    given Dataset into a NebulDataset. Note that the NebulDataset is
    transparent tothe Dataset custom methods. It just reimplement the
    __getitem__ method for accelerating the data loading part.

    Args:
        **nebuldata_kwargs (Dict): Extra parameters that must be passed to the
            NebulDataset. See the NebulDataset description for further
            information.

    Returns:
        a patched Dataset class which will create a NebulDataset instead of a
        base class instantiation.
    """

    def _inner_patch(cls: Type[Dataset]):
        def _new_new(cls: Type[Dataset], *args, **kwargs):
            self = object.__new__(cls)
            self.__init__(*args, **kwargs)
            return NebulDataset(self, **nebuldata_kwargs)

        cls.__new__ = _new_new
        return cls

    return _inner_patch
