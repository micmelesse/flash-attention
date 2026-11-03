"""ROCm backend selection and Dao-AI FA3 kernel access for the Hopper package."""

import importlib
import os
import warnings
from typing import Literal

AmdBackend = Literal["ck", "triton", "aiter"]


def select_amd_backend(is_rocm: bool | None = None) -> AmdBackend:
    triton = os.getenv("FLASH_ATTENTION_TRITON_AMD_ENABLE", "FALSE") == "TRUE"
    aiter = os.getenv("FLASH_ATTENTION_AITER_AMD_ENABLE", "FALSE") == "TRUE"
    if triton and aiter:
        raise ValueError(
            "Select only one of FLASH_ATTENTION_TRITON_AMD_ENABLE "
            "and FLASH_ATTENTION_AITER_AMD_ENABLE"
        )
    backend: AmdBackend = "aiter" if aiter else "triton" if triton else "ck"
    if backend == "aiter" and is_rocm is False:
        raise RuntimeError("FLASH_ATTENTION_AITER_AMD_ENABLE requires ROCm")
    if backend == "ck" and is_rocm:
        try:
            importlib.import_module("flash_attn_3._C")
        except ImportError:
            warnings.warn(
                "flash_attn_3._C (which has ROCm/HIP kernels) not found, "
                "falling back to Triton implementation"
            )
            return "triton"
    return backend


def get_flash_attn_3():
    from aiter.ops.triton._triton_kernels.attention.dao_ai import flash_attn_3

    return flash_attn_3
