import os.path
from collections import OrderedDict
from tempfile import mkdtemp
from threading import Thread, Lock
from typing import Any, List

import psutil
import torch
from torch.utils.data import Dataset

from nebulgym.data.base import BaseDataset


class WindowLoader:
    def __init__(
        self,
        window_size: int,
        max_loaded_size: int = None,
        max_writing_jobs: int = None,
    ):
        self._total_loaded_size = 0
        self._window_size = window_size
        self._max_loaded_size = max_loaded_size or (
            int(getattr(psutil.virtual_memory(), "available")) // 4
        )
        self._loaded_tensors = OrderedDict()
        self._cache = mkdtemp()
        self._tensor_map = {}
        self._writing_bus = []
        self._max_writing_jobs = max_writing_jobs or torch.get_num_threads()
        self._thread_lock = Lock()

    def _compute_size(self, inputs: Any):
        if isinstance(inputs, torch.Tensor):
            return inputs.element_size() * inputs.nelement()
        elif isinstance(inputs, dict):
            return sum(self._compute_size(val) for val in inputs.values())
        elif isinstance(inputs, int) or isinstance(inputs, float):
            return 32
        else:
            return sum(self._compute_size(val) for val in inputs)

    def _load_input(self, tensor_path: str):
        torch_input = torch.load(tensor_path)
        loaded_size = self._compute_size(torch_input)
        with self._thread_lock:
            self._total_loaded_size += loaded_size
        return torch_input

    def get(self, tensor_path: str):
        tensor_hash = hash(tensor_path)
        return self._loaded_tensors.get(tensor_hash)

    def load_new_batch(self, tensor_paths: List[str]):
        self._total_loaded_size = 0
        new_loaded_tensors = {
            hash(p): self._load_input(p)
            for p in tensor_paths
            if self._total_loaded_size <= self._max_loaded_size
        }
        return tensor_paths[len(new_loaded_tensors) :]  # noqa E203

    def store(self, torch_input: Any, idx: int):
        self._store_in_a_thread(torch_input, idx)
        while len(self._writing_bus) >= self._max_writing_jobs:
            thread = self._writing_bus.pop(0)
            thread.join()

    def _store(self, torch_input: Any, idx: int):
        input_name = f"data_idx_{idx}.pt"
        input_path = os.path.join(self._cache, input_name)
        torch.save(torch_input, input_path)
        self._tensor_map[idx] = input_path

    def _store_in_a_thread(self, torch_input: Any, idx: int):
        new_thread = Thread(
            target=self._store, args=(torch_input, idx), daemon=True
        )
        new_thread.start()
        self._writing_bus.append(new_thread)

    def join_all_writing_threads(self):
        while len(self._writing_bus) > 0:
            thread = self._writing_bus.pop(0)
            thread.join()

    def _schedule_multi_thread_data_loading(
        self, tensor_paths: List[str]
    ) -> Thread:
        first_thread = None
        for path in tensor_paths:
            hash_id = hash(path)
            t = Thread(
                target=self._load_data_in_thread,
                args=(hash_id, path),
                daemon=True,
            )
            if first_thread is None:
                first_thread = t
            self._loaded_tensors[hash_id] = t
            t.start()
        return first_thread

    def _load_data_in_thread(self, hashed_id: str, tensor_path: str):
        res = self._load_input(tensor_path)
        self._loaded_tensors[hashed_id] = res

    def _clean_memory(self):
        while self._total_loaded_size > self._max_loaded_size:
            _, tensor = self._loaded_tensors.popitem(last=False)
            if isinstance(tensor, Thread):
                raise RuntimeError(
                    f"Not enough memory allocated for loading the data! "
                    f"Please increase the maximum amount of memory. "
                    f"Given {self._max_loaded_size} B"
                )
            size = self._compute_size(tensor)
            self._total_loaded_size -= size

    def __getitem__(self, item):
        tensor_path = self._tensor_map[item]
        output = self.get(tensor_path)
        if output is None:
            ids = range(
                item, min(item + self._window_size, len(self._tensor_map))
            )
            tensor_paths = [self._tensor_map[idx] for idx in ids]
            item_thread = self._schedule_multi_thread_data_loading(
                tensor_paths
            )
            item_thread.join()
            return self.get(tensor_path)
        elif isinstance(output, Thread):
            output.join()
            return self.get(tensor_path)
        return output

    def len_cached_inputs(self):
        return len(self._tensor_map)


class NebulDataset(BaseDataset):
    def __init__(
        self,
        input_data: Dataset,
        preloaded_data: int = 1,
        max_memory_size: int = None,
    ):
        super(NebulDataset, self).__init__(input_data)
        self._loader = WindowLoader(
            window_size=preloaded_data, max_loaded_size=max_memory_size
        )

    @property
    def is_first_run(self):
        return len(self) != self._loader.len_cached_inputs()

    def __len__(self):
        return len(self._data)

    def __getitem__(self, item):
        if self.is_first_run:
            out = self._data[item]
            self._loader.store(out, item)
            if len(self) == self._loader.len_cached_inputs():
                self._loader.join_all_writing_threads()
        else:
            out = self._loader[item]
        return out
