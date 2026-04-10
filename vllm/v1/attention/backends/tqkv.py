# SPDX-License-Identifier: Apache-2.0
"""TQKV attention backend — re-exported from tqkv package.

The implementation has moved to tqkv.integrations.vllm.backend.
This shim maintains backward compatibility.
"""
from tqkv.integrations.vllm.backend import *  # noqa: F401,F403
