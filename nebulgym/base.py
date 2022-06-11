from enum import Enum


class NebulgymBackend(str, Enum):
    PYTORCH = "PYTORCH"
    TORCHSCRIPT = "TORCHSCRIPT"
    ONNXRUNTIME = "ONNXRUNTIME"
    RAMMER = "RAMMER"
