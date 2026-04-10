# Forwarding shim: tqa owns the implementation.
# Replace vllm-fork/vllm/v1/attention/ops/tqa/tq_page_metadata_cuda.py with this.
from tqkv.page_metadata import compute_page_metadata_cuda  # noqa: F401
