from typing import Dict, Type

from nebulgym.training_learners.model_engines.base import BaseEngine
from nebulgym.training_learners.model_engines.ort_engine import ORTEngine
from nebulgym.training_learners.model_engines.rammer_engine import RammerEngine
from nebulgym.training_learners.model_engines.torch_engine import (
    TorchEngine,
    TorchScriptEngine,
)
from nebulgym.base import NebulgymBackend

ENGINE_MAP: Dict[NebulgymBackend, Type[BaseEngine]] = {
    NebulgymBackend.PYTORCH: TorchEngine,
    NebulgymBackend.TORCHSCRIPT: TorchScriptEngine,
    NebulgymBackend.ONNXRUNTIME: ORTEngine,
    NebulgymBackend.RAMMER: RammerEngine,
}
