from __future__ import annotations

import ctypes

import numpy as np

from novasight.inference import tensorrt as tensorrt_module


def test_tensorrt_host_buffer_prefers_pinned_cuda_memory_when_available() -> None:
    class FakeCudaRuntime:
        cudaHostAllocDefault = 0

        def __init__(self) -> None:
            self.buffers: dict[int, ctypes.Array] = {}
            self.allocated: list[tuple[int, int]] = []
            self.freed: list[int] = []

        def cudaHostAlloc(self, nbytes: int, flags: int):
            buffer = ctypes.create_string_buffer(int(nbytes))
            pointer = ctypes.addressof(buffer)
            self.buffers[pointer] = buffer
            self.allocated.append((int(nbytes), int(flags)))
            return 0, pointer

        def cudaFreeHost(self, pointer: int) -> int:
            self.freed.append(int(pointer))
            self.buffers.pop(int(pointer), None)
            return 0

    cudart = FakeCudaRuntime()

    allocation = tensorrt_module._allocate_host_array(
        cudart,
        shape=(1, 3, 2, 2),
        dtype=np.float32,
    )

    assert allocation.pinned is True
    assert allocation.array.shape == (1, 3, 2, 2)
    assert allocation.array.dtype == np.float32
    assert allocation.array.ctypes.data == allocation.pointer
    assert cudart.allocated == [(allocation.array.nbytes, 0)]
    allocation.array.fill(1.25)

    allocation.release()

    assert cudart.freed == [allocation.pointer]


def test_tensorrt_host_buffer_falls_back_to_numpy_when_pinned_memory_unavailable() -> None:
    class FakeCudaRuntime:
        def cudaHostAlloc(self, _nbytes: int, _flags: int):
            return 1, 0

    allocation = tensorrt_module._allocate_host_array(
        FakeCudaRuntime(),
        shape=(1, 3, 2, 2),
        dtype=np.float16,
    )

    assert allocation.pinned is False
    assert allocation.array.shape == (1, 3, 2, 2)
    assert allocation.array.dtype == np.float16
    allocation.release()
